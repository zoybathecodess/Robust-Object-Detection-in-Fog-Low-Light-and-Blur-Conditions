"""Training loops: supervised, teacher-student self-training, evaluation, test-time adaptation."""
from __future__ import annotations

import copy
import math
import time

import torch

from data import strong_augment
from model import wrap_for_tta


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infinite(loader):
    while True:
        for b in loader:
            yield b


def _clean_boxes(boxes, labels, min_wh=2.0):
    keep = ((boxes[:, 2] - boxes[:, 0]) > min_wh) & ((boxes[:, 3] - boxes[:, 1]) > min_wh)
    return {"boxes": boxes[keep], "labels": labels[keep]}


def make_pseudo(outputs, thr):
    """Keep confident teacher detections as pseudo ground truth."""
    res = []
    for o in outputs:
        k = o["scores"] >= thr
        res.append(_clean_boxes(o["boxes"][k].detach(), o["labels"][k].detach()))
    return res


class AmpStep:
    def __init__(self, opt, params, amp, clip=10.0):
        self.opt, self.params, self.clip = opt, list(params), clip
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def __call__(self, loss):
        if not torch.isfinite(loss):
            return False
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        return True


def _cosine_warmup(total, warmup=200):
    return lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))


def _sup_loss(model, imgs, tgts, device):
    imgs = [i.to(device) for i in imgs]
    tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
    return sum(model(imgs, tgts).values())


# --------------------------------------------------------------------------- supervised
def train_supervised(model, loader, epochs, lr, device, log_every=20):
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _cosine_warmup(total))
    step_fn = AmpStep(opt, params, amp=device.type == "cuda")
    step, t0 = 0, time.time()
    for ep in range(epochs):
        for imgs, tgts in loader:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = _sup_loss(model, imgs, tgts, device)
            step_fn(loss)
            sched.step()
            step += 1
            if step % log_every == 0:
                print(f"[sup] ep {ep + 1}/{epochs} step {step}/{total} loss {loss.item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return model


# --------------------------------------------------------------------------- self-training
@torch.no_grad()
def ema_update(teacher, student, decay):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(decay).add_(ps.detach(), alpha=1 - decay)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


def selftrain(student, labeled_loader, unlabeled_loader, device, steps, lr=0.005,
              thr=0.7, lam=1.0, ema_decay=0.999, log_every=20):
    """Mean-teacher pseudo-labelling.

    * supervised loss on labeled (clean + synthetically degraded) batches
    * teacher (EMA of student) labels the weakly-augmented real degraded image;
      confident boxes become targets for the student on a strongly-augmented view
    Returns the teacher (usually the better model)."""
    student.to(device).train()
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _cosine_warmup(steps, warmup=min(100, steps // 5 + 1)))
    step_fn = AmpStep(opt, params, amp=device.type == "cuda")
    lab, unl = infinite(labeled_loader), infinite(unlabeled_loader)
    ramp = max(1, steps // 10)
    t0 = time.time()
    for step in range(1, steps + 1):
        imgs, tgts = next(lab)
        uimgs = [u.to(device) for u in next(unl)]
        student.train()
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss_s = _sup_loss(student, imgs, tgts, device)
            with torch.no_grad():
                teacher.eval()
                pseudo = make_pseudo(teacher(uimgs), thr)
            n_pl = sum(len(p["boxes"]) for p in pseudo)
            loss_u = torch.zeros((), device=device)
            if n_pl > 0:
                strong = [strong_augment(u) for u in uimgs]
                loss_u = sum(student(strong, pseudo).values())
            loss = loss_s + lam * min(1.0, step / ramp) * loss_u
        step_fn(loss)
        sched.step()
        ema_update(teacher, student, ema_decay)
        if step % log_every == 0:
            print(f"[self] step {step}/{steps} sup {loss_s.item():.3f} unsup {float(loss_u):.3f} "
                  f"pseudo-boxes {n_pl} ({time.time() - t0:.0f}s)", flush=True)
    return teacher


# --------------------------------------------------------------------------- evaluation
def _metric():
    from torchmetrics.detection import MeanAveragePrecision
    return MeanAveragePrecision(iou_type="bbox")


def _summ(metric):
    r = metric.compute()
    return {"mAP": round(r["map"].item() * 100, 2), "mAP50": round(r["map_50"].item() * 100, 2)}


@torch.no_grad()
def evaluate(model, loader, device):
    metric = _metric()
    model.to(device).eval()
    for imgs, tgts in loader:
        outs = model([i.to(device) for i in imgs])
        metric.update([{k: v.cpu() for k, v in o.items()} for o in outs],
                      [{"boxes": t["boxes"], "labels": t["labels"]} for t in tgts])
    return _summ(metric)


def tta_evaluate(model, loader, device, lr=1e-3, thr=0.8, episodic=False):
    """Online test-time adaptation. For every incoming batch:
       1. detect with the current model, keep confident boxes as pseudo-labels
       2. one gradient step on a strongly-augmented view, updating ONLY the FPN
          scale/shift parameters (FeatureAffine)
       3. re-detect with the adapted model and score those predictions
    episodic=True resets the adapter after every batch (no accumulation)."""
    model = wrap_for_tta(copy.deepcopy(model)).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    ad = model.backbone.adapt_parameters()
    for p in ad:
        p.requires_grad_(True)
    init = [p.detach().clone() for p in ad]
    opt = torch.optim.SGD(ad, lr=lr, momentum=0.9)
    metric = _metric()
    for imgs, tgts in loader:
        imgs = [i.to(device) for i in imgs]
        model.eval()
        with torch.no_grad():
            pseudo = make_pseudo(model(imgs), thr)
        if sum(len(p["boxes"]) for p in pseudo):
            model.train()  # FrozenBN backbone => train mode only changes proposal/ROI sampling
            loss = sum(model([strong_augment(i) for i in imgs], pseudo).values())
            if torch.isfinite(loss):
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            outs = model(imgs)
        metric.update([{k: v.cpu() for k, v in o.items()} for o in outs],
                      [{"boxes": t["boxes"], "labels": t["labels"]} for t in tgts])
        if episodic:
            with torch.no_grad():
                for p, p0 in zip(ad, init):
                    p.copy_(p0)
    return _summ(metric)
