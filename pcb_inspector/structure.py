"""Second opinion for pictures that were not taken/rendered identically.

Colour comparison cannot see a dark part that was replaced by bare dark board with
two shiny pads.  What does change there is the local structure: the picture of the part
(its edges, pins, markings) can no longer be found in the inspection picture anywhere
near the same place.  Slide every reference patch over a small window of the
inspection picture; patches whose best normalised correlation stays low are changed.
"""
import cv2
import numpy as np


MIN_AGREEMENT = .80
STRICT_BELOW = .90
DIFF_MIN = 24


def structure_boxes(reference, aligned, valid):
    g1 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g1.shape
    short = min(H, W)
    cs = max(24, int(round(.065 * short)))
    st = max(4, cs // 4)
    sh = max(5, int(round(.02 * short)))
    cells = []
    for y in range(0, H - cs + 1, st):
        for x in range(0, W - cs + 1, st):
            if not valid[y:y + cs, x:x + cs].all():
                continue
            p = g1[y:y + cs, x:x + cs]
            if p.std() < 5:
                continue
            ys, xs = max(0, y - sh), max(0, x - sh)
            win = g2[ys:min(H, y + cs + sh), xs:min(W, x + cs + sh)]
            cells.append((x, y, float(cv2.matchTemplate(win, p, cv2.TM_CCOEFF_NORMED).max())))
    if len(cells) < 20:
        return []
    # two renders of the same board never correlate perfectly everywhere: judge each
    # patch against how well the rest of this very pair agrees
    median = float(np.median([c[2] for c in cells]))
    structure_boxes.last_median = median
    if median < MIN_AGREEMENT:
        return []          # different render/photo throughout: structure says nothing useful
    thresh = max(.15, min(.5, median - .32))
    if median < STRICT_BELOW:
        thresh = min(thresh, .35)     # only clear-cut changes when the pair differs a lot overall
    # seeds = clear-cut changes; grow each seed over neighbouring patches that also
    # correlate poorly, so the box covers the whole part and not only its most different edge
    seed = np.zeros((H, W), np.uint8)
    weak = np.zeros((H, W), np.uint8)
    for x, y, v in cells:
        sl = (slice(y + cs // 4, y + cs - cs // 4), slice(x + cs // 4, x + cs - cs // 4))
        if v < thresh:
            seed[sl] = 1
        if v < min(.62, thresh + .12):
            weak[sl] = 1
    nw, labw = cv2.connectedComponents(weak, connectivity=8)
    flag = seed.copy()
    for wid in set(np.unique(labw[seed > 0])) - {0}:
        grown = labw == wid
        ys, xs = np.where(grown)
        sy, sx = np.where(grown & (seed > 0))
        gbox = (xs.max() - xs.min()) * (ys.max() - ys.min())
        sbox = max(1, (sx.max() - sx.min()) * (sy.max() - sy.min()))
        if gbox <= 3.0 * sbox and (xs.max() - xs.min()) < .3 * W and (ys.max() - ys.min()) < .3 * H:
            flag[grown] = 1            # same part: take all of it
    flag = cv2.morphologyEx(flag, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(flag, connectivity=8)
    diff = cv2.absdiff(cv2.GaussianBlur(g1, (0, 0), 2), cv2.GaussianBlur(g2, (0, 0), 2))
    def edge(g, x, y, w, h):
        q = g[y:y + h, x:x + w]
        return float(np.hypot(cv2.Sobel(q, cv2.CV_32F, 1, 0), cv2.Sobel(q, cv2.CV_32F, 0, 1)).mean())

    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < .35 * cs * cs or w > .30 * W or h > .30 * H:
            continue
        if max(w, h) > 3.2 * min(w, h):
            continue           # a long thin strip is a printed line / edge, not a part
        # the flagged core is the middle of each patch: grow it back to the patch
        gx = cs // 6
        x0, y0 = max(0, x - gx), max(0, y - gx)
        x1, y1 = min(W, x + w + gx), min(H, y + h + gx)
        core = (lab == i) & (seed > 0)
        sc = float(np.percentile(diff[core], 75)) if core.any() else float(np.percentile(diff[y0:y1, x0:x1], 75))
        if sc < DIFF_MIN:
            continue           # looks the same, only its fine detail differs
        if edge(g2, x0, y0, x1 - x0, y1 - y0) < edge(g1, x0, y0, x1 - x0, y1 - y0):
            continue           # the inspection picture shows LESS detail here: the part
                               # is still there, just drawn plainer (handled by the colour test)
        out.append((sc, (x0, y0, x1 - x0, y1 - y0)))
    out.sort(key=lambda t: -t[0])
    cap = 6 if median >= STRICT_BELOW else 4      # a short list of the surest changes
    return [b for _, b in out[:cap]]
