"""One general missing-part detector (v1.0). Works on any board colour.

Replaces the old stack of per-colour routes (green / black / red / AI boxes).
No component classification, no AI model: it only answers
"what physical object in the reference is no longer in the inspection image?".

Pipeline (all automatic, nothing to tune in the GUI):
 1. work at <= 900 px so thresholds mean the same thing for any photo size
 2. fit per-channel gain/offset (robust) to cancel lighting / white balance
 3. shift-tolerant colour difference (min over a +-2 px window) -> alignment
    noise at edges disappears, a removed body (large flat area) does not
 4. hysteresis threshold: strong seeds grow into the weaker connected region
 5. every blob must pass physical checks: solid (not a thin trace line),
    thick enough, strong, and its structure must really differ between images
 6. blobs that are reference texture lost (-> "Likely missing") are separated
    from blobs that gained texture (-> "Needs review": added / changed part)
"""
from dataclasses import dataclass

import cv2
import numpy as np

WORK_MAX = 900
W_L, W_AB = .5, 2.0
LOW_MULT, LOW_MIN, LOW_MAX = 6.0, 11.0, 20.0


@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int
    strength: float        # mean colour distance / threshold
    structure_loss: float  # 1 = reference structure completely gone
    status: str            # 'Likely missing' | 'Needs review'
    reason: str
    mask: object = None    # work-resolution pixel mask (internal)


def _robust_normalise(ref, ins, valid):
    """Map ins onto ref's brightness/colour curve per channel (gain, offset, then a
    smooth non-linear correction) using only pixels that did not change."""
    out = np.empty(ins.shape, np.float32)
    rb = cv2.GaussianBlur(ref, (0, 0), 2).astype(np.float32)
    ib = cv2.GaussianBlur(ins, (0, 0), 2).astype(np.float32)
    ys, xs = np.nonzero(valid)
    if len(ys) > 80000:
        pick = np.random.default_rng(0).choice(len(ys), 80000, replace=False)
        ys, xs = ys[pick], xs[pick]
    for c in range(3):
        x = ib[ys, xs, c]
        y = rb[ys, xs, c]
        ok = (x > 4) & (x < 250) & (y > 4) & (y < 250)      # ignore clipped pixels
        if ok.sum() < 200:
            ok = np.ones_like(x, bool)
        x, y = x[ok], y[ok]
        wgt = np.ones_like(x)
        a, b = 1.0, 0.0
        for _ in range(8):
            sw = wgt.sum()
            if sw < 50:
                break
            mx, my = (wgt * x).sum() / sw, (wgt * y).sum() / sw
            var = (wgt * (x - mx) ** 2).sum() / sw
            if var < 1e-3:
                break
            a = float(np.clip((wgt * (x - mx) * (y - my)).sum() / sw / var, .6, 1.6))
            b = float(my - a * mx)
            res = np.abs(a * x + b - y)
            sc = max(3.0, 1.4826 * float(np.median(res)) * 2.0)
            wgt = 1.0 / (1.0 + (res / sc) ** 2) * (res < 4 * sc)
        lin = a * x + b
        # smooth non-linear residual: median (y - lin) in 12 intensity bins
        edges = np.linspace(0, 255, 13)
        centres, corr = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (lin >= lo) & (lin < hi) & (wgt > .5)
            if m.sum() > 150:
                centres.append(float(np.median(lin[m])))
                corr.append(float(np.clip(np.median((y - lin)[m]), -25, 25)))
        full = a * ins[:, :, c].astype(np.float32) + b
        if len(centres) >= 3:
            full = full + np.interp(full, centres, corr)
        out[:, :, c] = full
    return np.clip(out, 0, 255).astype(np.uint8)


def _tolerant_distance(ref, ins, radius):
    """min over +-radius shifts of the LAB colour distance (float32, HxW)."""
    lab_r = cv2.cvtColor(cv2.GaussianBlur(ref, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_i = cv2.cvtColor(cv2.GaussianBlur(ins, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab_r.shape[:2]
    pad = radius
    padded = cv2.copyMakeBorder(lab_i, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius + 1:
                continue
            sh = padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            d = lab_r - sh
            # lightness counts a bit less than colour: shadows are not parts
            dist = np.sqrt(W_L * d[:, :, 0] ** 2 + W_AB * d[:, :, 1] ** 2 + W_AB * d[:, :, 2] ** 2)
            best = dist if best is None else np.minimum(best, dist)
    return best


def _ncc(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-6 else 1.0


def detect(reference: np.ndarray, aligned: np.ndarray, valid: np.ndarray, debug=None, wide=False, texture=False):
    """Return (list[Blob] in ORIGINAL pixel coordinates, distance map uint8, mask uint8)."""
    H, W = reference.shape[:2]
    scale = min(1.0, WORK_MAX / max(H, W))
    if scale < 1.0:
        size = (round(W * scale), round(H * scale))
        ref = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        ins = cv2.resize(aligned, size, interpolation=cv2.INTER_AREA)
        val = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    else:
        ref, ins, val = reference, aligned, valid
    h, w = ref.shape[:2]
    # Work-resolution is <= 900 px, but boards photographed small stay small:
    # tolerance in pixels follows the image, not a hard number.
    radius = 2 if min(h, w) >= 300 else 1

    # exclude invalid (warped-in border) area and a thin frame at the border
    margin = max(3, round(min(h, w) * .008))
    safe = cv2.erode(val.astype(np.uint8) * 255,
                     cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1))) > 0

    ins_n = _robust_normalise(ref, ins, safe)
    dist = _tolerant_distance(ref, ins_n, radius)
    # two different photographs/renders: edges never line up perfectly, so widen
    # the shift tolerance (filled areas of a removed part still survive it)
    if wide and radius >= 2:
        radius = 3
        dist = _tolerant_distance(ref, ins_n, radius)
    # texture channel: a part that is black-on-black (or vanished chip pins) leaves
    # no colour change but its fine structure (pins/markings) appears or disappears
    if texture:
        def energy(img):
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), max(3.0, .006 * min(h, w)))
        er_, ei_ = energy(ref), energy(ins_n)
        # shift tolerance: compare against the local min/max of the other image
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        ei_hi, ei_lo = cv2.dilate(ei_, k), cv2.erode(ei_, k)
        diff = np.maximum(er_ - ei_hi, ei_lo - er_)
        rel = np.clip(diff, 0, None) / (np.maximum(er_, ei_) + 25.0)
        dist = np.maximum(dist, rel * 90.0)
    dist[~safe] = 0

    inside = dist[safe]
    if inside.size == 0:
        return [], np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    med = float(np.median(inside))
    mad = float(np.median(np.abs(inside - med))) * 1.4826
    low = float(np.clip(med + LOW_MULT * mad, LOW_MIN, LOW_MAX))
    levels = [low, low * 1.35, low * 1.8, low * 2.4]

    gray_r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_i = cv2.cvtColor(ins_n, cv2.COLOR_BGR2GRAY)
    # Clean pair (same file edited / pixel-aligned): tiny parts are believable.
    # Two different photographs: small SMD blobs are mostly rendering/lighting
    # differences, so only part-sized regions are reported.
    p90 = float(np.percentile(inside, 90))
    noisy = p90 > 6.0
    min_area = max(40, round(h * w * (.0010 if noisy else .00016)))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blobs, kept_mask = [], np.zeros((h, w), np.uint8)

    def log(cx, cy, area, why):
        if debug is not None:
            debug.append((round(cx / scale), round(cy / scale), area, why))

    level_ref = [0]

    def check(comp, x, y, bw, bh, area):
        """-> ('ok', Blob) | ('split', None) | ('drop', None)"""
        cx, cy = x + bw / 2, y + bh / 2
        if area < min_area or bw < 5 or bh < 5:
            return 'drop', None
        touches = (x <= margin + 1 or y <= margin + 1 or
                   x + bw >= w - margin - 1 or y + bh >= h - margin - 1)
        if bw * bh > .22 * h * w or (touches and area >= .02 * h * w):
            log(cx, cy, area, 'too big -> split')
            return 'split', None
        solidity = area / float(bw * bh)
        aspect = max(bw, bh) / float(min(bw, bh))
        if solidity < .30 or aspect > 7:
            log(cx, cy, area, f'merged shape sol{solidity:.2f} asp{aspect:.1f} -> split')
            return 'split', None
        # a real removed part is uniformly different inside; a patchy blob that
        # merely strings together many small differences is split into its cores
        core = float(np.mean(dist[comp] > 1.5 * levels[0]))
        if area > .004 * h * w and core < .30 and level_ref[0] + 1 < len(levels):
            log(cx, cy, area, f'patchy core{core:.2f} -> split')
            return 'split', None
        thick = float(cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3).max())
        if thick < 2.4:
            log(cx, cy, area, f'thin {thick:.1f}')
            return 'drop', None
        if min(bw, bh) < .04 * min(h, w) and aspect > 4.0:
            log(cx, cy, area, 'thin strip (edge/shadow)')
            return 'drop', None
        mean_d = float(dist[comp].mean())
        strength = mean_d / low
        if mean_d < 17:
            log(cx, cy, area, f'weak p90={np.percentile(dist[comp],90):.0f} str{strength:.2f} D{mean_d:.0f}')
            return 'drop', None
        pad = 3
        xa, ya = max(0, x - pad), max(0, y - pad)
        xb, yb = min(w, x + bw + pad), min(h, y + bh + pad)
        pr, pi = gray_r[ya:yb, xa:xb], gray_i[ya:yb, xa:xb]
        cm = comp[ya:yb, xa:xb]
        if min(pr.shape) >= 8:
            best = -1.0
            core = pr[2:-2, 2:-2]
            pip = cv2.copyMakeBorder(pi, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
            for dy in range(0, 5):
                for dx in range(0, 5):
                    best = max(best, _ncc(core, pip[dy:dy + core.shape[0], dx:dx + core.shape[1]]))
            if best > .82 and strength < 2.6:
                log(cx, cy, area, f'same structure ncc{best:.2f}')
                return 'drop', None
        sr = float(np.std(pr[cm])) if cm.any() else 0.
        si = float(np.std(pi[cm])) if cm.any() else 0.
        er = float((cv2.Canny(pr, 50, 120) > 0).mean())
        ei = float((cv2.Canny(pi, 50, 120) > 0).mean())
        loss = float(np.clip(.5 * (er - ei) / max(er, .02) + .5 * (sr - si) / max(sr, 6.), -1, 1))
        # does the inspection patch look like the board around it (part removed)?
        ring = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        ring = (ring > 0) & ~cv2.dilate(comp.astype(np.uint8), k5).astype(bool) & safe
        if ring.sum() > 20:
            rc = np.median(cv2.cvtColor(ins_n, cv2.COLOR_BGR2LAB)[ring].reshape(-1, 3), axis=0).astype(np.float32)
            lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[comp].astype(np.float32)
            lab_i = cv2.cvtColor(ins_n, cv2.COLOR_BGR2LAB)[comp].astype(np.float32)
            d_ref = float(np.linalg.norm(np.median(lab_r, axis=0) - rc))
            d_ins = float(np.linalg.norm(np.median(lab_i, axis=0) - rc))
        else:
            d_ref, d_ins = 1., 1.
        if not texture and max(d_ref, d_ins) < 18:
            log(cx, cy, area, f'no contrast to surroundings dRef{d_ref:.0f} dIns{d_ins:.0f}')
            return 'drop', None
        # texture loss alone only counts when the part has no colour contrast to the
        # board (black part on black board); otherwise the part must look like board now
        gone = (d_ins < .6 * d_ref and d_ref > 10) or (loss > .2 and d_ref < 25)
        # weak colour change is only believable when the part clearly turned
        # into bare board (dark IC on dark-green mask has low contrast)
        # a small part swapped for bare pads: weak on average but with a hot, compact core
        hot_core = area >= 450 and d_ref >= 35 and float(np.percentile(dist[comp], 90)) >= 32 and solidity >= .45
        if not texture and mean_d < 27 and not hot_core and not (gone and area >= 300 and solidity >= .40 and thick >= 5):
            log(cx, cy, area, f'weak-and-not-bare p90={np.percentile(dist[comp],90):.0f} D{mean_d:.0f} dRef{d_ref:.0f} dIns{d_ins:.0f} sol{solidity:.2f}')
            return 'drop', None
        if strength < 1.45 and not gone:
            log(cx, cy, area, f'weak str{strength:.2f} dRef{d_ref:.0f} dIns{d_ins:.0f} sol{solidity:.2f}')
            return 'drop', None
        tiny = area < .0008 * h * w
        if tiny and not (strength >= 2.2 or (mean_d >= 30 and d_ref >= 60 and d_ins < .7 * d_ref)):
            log(cx, cy, area, f'tiny-weak D{mean_d:.0f} dRef{d_ref:.0f} dIns{d_ins:.0f}')
            return 'drop', None
        status = 'Likely missing' if gone else 'Needs review'
        reason = (f'colour change {mean_d:.0f} (noise limit {low:.0f}); {round(bw/scale)}x{round(bh/scale)}px; '
                  + ('region now looks like bare board' if gone else 'region changed, still differs from bare board'))
        log(cx, cy, area, f'KEPT str{strength:.2f} D{mean_d:.0f} sol{solidity:.2f} thick{thick:.1f} '
                          f'loss{loss:+.2f} dRef{d_ref:.0f} dIns{d_ins:.0f} {status}')
        return 'ok', Blob(round(x / scale), round(y / scale), round(bw / scale), round(bh / scale),
                          strength, loss, status, reason, comp)

    def recurse(parent, level):
        """Return the list of valid blobs found inside `parent` at threshold `level`."""
        level_ref[0] = level
        weak = ((dist > levels[level]) & parent).astype(np.uint8)
        weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, k3)
        weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, k5)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
        found = []
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            comp = labels == i
            level_ref[0] = level
            verdict, blob = check(comp, x, y, bw, bh, area)
            if verdict == 'ok':
                # a big accepted blob may be several neighbouring parts glued
                # together by weak differences: prefer its separate cores
                if area > .008 * h * w and level + 1 < len(levels):
                    kids = recurse(comp, level + 1)
                    level_ref[0] = level
                    if len(kids) >= 2:
                        found.extend(kids)
                        continue
                found.append(blob)
            elif (verdict == 'split' or (verdict == 'drop' and area > .008 * h * w)) and level + 1 < len(levels):
                found.extend(recurse(comp, level + 1))
        return found

    for blob in recurse(safe, 0):
        blobs.append(blob)
        kept_mask[blob.mask] = 255

    # merge boxes that overlap strongly (same part broken in two)
    blobs = _merge_overlapping(blobs, (H, W), scale)
    for blob in blobs:
        _tighten(blob, dist, scale, (H, W))
    mask_full = cv2.resize(kept_mask, (W, H), interpolation=cv2.INTER_NEAREST) if scale < 1 else kept_mask
    dist_full = np.clip(dist * (255. / 80.), 0, 255).astype(np.uint8)
    if scale < 1:
        dist_full = cv2.resize(dist_full, (W, H), interpolation=cv2.INTER_LINEAR)
    blobs.sort(key=lambda b: (0 if b.status == 'Likely missing' else 1, -b.strength))
    return blobs, dist_full, mask_full


def _tighten(blob, dist, scale, shape):
    """The mask grows out to the faint fringe where the two pictures are not perfectly
    registered. Put the box on the part itself: the pixels that really differ."""
    m = blob.mask
    d = dist[m]
    if d.size < 30:
        return
    cut = max(.5 * float(np.percentile(d, 90)), .6 * float(d.mean()))
    core = (m & (dist >= cut)).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n < 2:
        return
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= .08 * core.sum()]
    if not keep:
        return
    x0 = min(st[i, 0] for i in keep); y0 = min(st[i, 1] for i in keep)
    x1 = max(st[i, 0] + st[i, 2] for i in keep); y1 = max(st[i, 1] + st[i, 3] for i in keep)
    if (x1 - x0) * (y1 - y0) < .35 * blob.w * scale * blob.h * scale:
        return                       # would shrink to a sliver: keep the original box
    H, W = shape
    blob.x, blob.y = int(round(x0 / scale)), int(round(y0 / scale))
    blob.w = min(W - blob.x, int(round((x1 - x0) / scale)))
    blob.h = min(H - blob.y, int(round((y1 - y0) / scale)))


def _merge_overlapping(blobs, shape, scale):
    """Join fragments of ONE part: pixel masks closer than a few px AND the joint box stays compact."""
    gap = max(3, round(.006 * max(shape) * scale))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    blobs = list(blobs)
    changed = True
    while changed:
        changed = False
        grown = [cv2.dilate(b.mask.astype(np.uint8), ker) > 0 for b in blobs]
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                a, b = blobs[i], blobs[j]
                if not (grown[i] & b.mask).any():
                    continue
                x1, y1 = min(a.x, b.x), min(a.y, b.y)
                x2, y2 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
                if (x2 - x1) * (y2 - y1) > 1.8 * (a.w * a.h + b.w * b.h):
                    continue
                keep = a if a.strength >= b.strength else b
                status = 'Likely missing' if 'Likely missing' in (a.status, b.status) else keep.status
                blobs[i] = Blob(x1, y1, x2 - x1, y2 - y1, keep.strength, keep.structure_loss,
                                status, keep.reason, a.mask | b.mask)
                del blobs[j]
                changed = True
                break
            if changed:
                break
    return blobs
