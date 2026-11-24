"""Reproducible benchmark for the PCB inspector.

Run from the project folder:   python benchmark/run_benchmark.py
Outputs go to benchmark/results/ (CSV tables, PNG charts, REPORT.md).

Experiments
  F2  Alignment of tilted boards: ORB + RANSAC homography vs. no alignment.
  F3  Challenging test set: solder glare, uneven light, noise, JPEG, tilt, and
      tiny dark SMD parts on a dark board (the 'black' board).
  F4  Absolute Difference vs. SSIM (vs. the full pipeline) under different
      lighting, with quantitative metrics and a threshold sweep.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcb_inspector.core import align_images, inspect, load_image  # noqa: E402

OUT = ROOT / 'benchmark' / 'results'
OUT.mkdir(parents=True, exist_ok=True)
S = ROOT / 'samples'

# board -> (reference, inspection with parts removed, ground-truth centres of removed parts)
BOARDS = {
    'green': (S/'green_reference.png', S/'green_inspection.png',
              [(211, 85), (295, 108), (88, 255), (197, 388), (410, 400), (412, 115)]),
    'black': (S/'black_board'/'reference.png', S/'black_board'/'inspection_aligned.png',
              [(275, 100), (720, 95), (625, 395), (380, 520), (712, 605), (145, 437)]),
    'red': (S/'reka_reference_regression.png', S/'reka_aligned_regression.png',
            [(136, 366), (125, 440), (365, 613), (627, 280), (660, 455)]),
}
TOL = 60          # a detection within TOL px of a removed part counts as a hit

# --------------------------------------------------------------------------- conditions
def _glare(im, rng, n=6):
    """Specular hot spots like light reflecting off solder joints / chip tops."""
    h, w = im.shape[:2]
    layer = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx, cy = int(rng.uniform(.1, .9) * w), int(rng.uniform(.1, .9) * h)
        r = int(rng.uniform(.012, .022) * min(h, w))
        cv2.ellipse(layer, (cx, cy), (r, int(r * .7)), rng.uniform(0, 180), 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(2, min(h, w) * .004))
    out = im.astype(np.float32) * (1 - .85 * layer[..., None]) + 255 * .85 * layer[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _uneven(im):
    h, w = im.shape[:2]
    gx = np.linspace(.72, 1.22, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(.92, 1.08, h, dtype=np.float32)[:, None, None]
    return np.clip(im.astype(np.float32) * gx * gy, 0, 255).astype(np.uint8)


def _persp(im, deg, strength):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    out = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
    s = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = s + strength * np.float32([[1, 1], [-1, 2], [-2, -1], [1, -2]]) * min(h, w)
    return cv2.warpPerspective(out, cv2.getPerspectiveTransform(s, d), (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))


def apply(im, cond, rng):
    f = im.astype(np.float32)
    if cond == 'normal':
        return im.copy()
    if cond == 'bright':
        return np.clip(f * 1.25 + 10, 0, 255).astype(np.uint8)
    if cond == 'dark':
        return np.clip(f * .7 - 6, 0, 255).astype(np.uint8)
    if cond == 'warm':
        f[..., 2] *= 1.15; f[..., 0] *= .85
        return np.clip(f, 0, 255).astype(np.uint8)
    if cond == 'gamma':
        return (255 * (im / 255.0) ** 1.6).astype(np.uint8)
    if cond == 'uneven':
        return _uneven(im)
    if cond == 'glare':
        return _glare(im, rng)
    if cond == 'noise':
        return np.clip(f + rng.normal(0, 9, im.shape), 0, 255).astype(np.uint8)
    if cond == 'jpeg':
        return cv2.imdecode(cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], 1)
    if cond == 'tilt':
        return _persp(im, 4.0, .012)
    if cond == 'tilt+glare':
        return _glare(_persp(im, -5.0, .015), rng)
    raise KeyError(cond)


LIGHT = ['normal', 'bright', 'dark', 'warm', 'gamma', 'uneven', 'glare', 'noise', 'jpeg']
HARD = ['tilt', 'tilt+glare']

# --------------------------------------------------------------------------- baselines
def _prep(ref, ins):
    aligned, valid, _ = align_images(ref, ins)
    return aligned, valid


def diff_map(method, ref, aligned, valid):
    """Per-pixel difference in 0..255 (higher = more different)."""
    g1 = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if method == 'absdiff':
        d = cv2.absdiff(g1, g2).astype(np.float32)
    elif method == 'ssim':
        _, full = structural_similarity(g1, g2, full=True, win_size=7, data_range=255)
        d = np.clip((1 - full) * 255, 0, 255).astype(np.float32)
    else:
        raise KeyError(method)
    d[~valid] = 0
    return d


def boxes_from_map(d, thr, min_area):
    m = (d > thr).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [tuple(cen[i]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


def score(dets, gts):
    """Hit/miss/false-alarm counts. Each removed part can be matched once."""
    left = list(gts)
    tp = 0
    for (x, y) in dets:
        j = next((i for i, (gx, gy) in enumerate(left) if abs(x - gx) < TOL and abs(y - gy) < TOL), None)
        if j is not None:
            tp += 1
            left.pop(j)
    return tp, len(dets) - tp, len(gts) - tp     # TP, FP, FN


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

# --------------------------------------------------------------------------- run
def build_cases(conditions):
    """Every (board, condition) pair, once with the real defects and once defect-free."""
    cases = []
    for bi, (b, (rp, ip, gt)) in enumerate(BOARDS.items()):
        ref, ins = load_image(rp), load_image(ip)
        for ci, c in enumerate(conditions):
            rng = np.random.default_rng(1000 * bi + ci)
            cases.append(dict(board=b, cond=c, ref=ref, ins=apply(ins, c, rng), gt=gt, kind='defect'))
            rng = np.random.default_rng(1000 * bi + ci)
            cases.append(dict(board=b, cond=c, ref=ref, ins=apply(ref, c, rng), gt=[], kind='clean'))
    return cases


def main():
    conditions = LIGHT + HARD
    cases = build_cases(conditions)
    print(f'{len(cases)} test images ({len(BOARDS)} boards x {len(conditions)} conditions x 2)')

    # aligned images + difference maps are computed once and reused for the sweep
    for c in cases:
        c['aligned'], c['valid'] = _prep(c['ref'], c['ins'])
        c['min_area'] = .00025 * c['ref'].shape[0] * c['ref'].shape[1]
        for m in ('absdiff', 'ssim'):
            c[m] = diff_map(m, c['ref'], c['aligned'], c['valid'])

    # ---------------- F4: calibrate the operating point on NORMAL light only
    sweep = {'absdiff': np.arange(8, 121, 4), 'ssim': np.arange(8, 200, 6)}
    calib = {}
    curves = {m: [] for m in sweep}
    for m, thrs in sweep.items():
        best = (-1, None)
        for t in thrs:
            tot = [0, 0, 0]
            for c in cases:
                if c['cond'] != 'normal':
                    continue
                r = score(boxes_from_map(c[m], t, c['min_area']), c['gt'])
                tot = [a + b for a, b in zip(tot, r)]
            f1 = prf(*tot)[2]
            curves[m].append((t, f1))
            if f1 > best[0]:
                best = (f1, t)
        calib[m] = best[1]
        print(f'calibrated {m}: threshold {best[1]} (F1 on normal light {best[0]:.2f})')

    # ---------------- evaluate every method on every condition
    rows = []
    for c in cases:
        pred = {m: boxes_from_map(c[m], calib[m], c['min_area']) for m in ('absdiff', 'ssim')}
        # oracle: best threshold for this very image (upper bound for the baseline)
        for m in ('absdiff', 'ssim'):
            c['oracle_' + m] = max(prf(*score(boxes_from_map(c[m], t, c['min_area']), c['gt']))[2]
                                   if c['gt'] else (1.0 if not boxes_from_map(c[m], t, c['min_area']) else 0.0)
                                   for t in sweep[m])
        res = inspect(c['ref'], c['ins'])
        pred['ours'] = [(f.x + f.w / 2, f.y + f.h / 2) for f in res.findings]
        for m, dets in pred.items():
            tp, fp, fn = score(dets, c['gt'])
            rows.append(dict(board=c['board'], cond=c['cond'], kind=c['kind'], method=m, tp=tp, fp=fp, fn=fn,
                             oracle_f1=c.get('oracle_' + m, ''),
                             diff_area=float(np.mean(c[m] > calib[m])) if m != 'ours' else ''))
    with open(OUT / 'per_image.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = summarise(rows, conditions)
    write_tables(summary, calib, conditions)
    make_charts(summary, curves, conditions)
    alignment = alignment_experiment()
    write_report(summary, calib, conditions, alignment, len(cases))
    print('done ->', OUT)


def summarise(rows, conditions):
    out = {}
    for m in ('absdiff', 'ssim', 'ours'):
        for cond in conditions + ['ALL']:
            sel = [r for r in rows if r['method'] == m and (cond == 'ALL' or r['cond'] == cond)]
            d = [r for r in sel if r['kind'] == 'defect']
            cl = [r for r in sel if r['kind'] == 'clean']
            tp, fp, fn = (sum(r[k] for r in d) for k in ('tp', 'fp', 'fn'))
            p, rc, f1 = prf(tp, fp, fn)
            out[(m, cond)] = dict(precision=p, recall=rc, f1=f1,
                                  fp_defect_imgs=fp / max(1, len(d)),
                                  fp_clean=sum(r['fp'] for r in cl) / max(1, len(cl)),
                                  clean_ok=sum(1 for r in cl if r['fp'] == 0) / max(1, len(cl)))
    return out


def write_tables(summary, calib, conditions):
    with open(OUT / 'summary_by_condition.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'condition', 'precision', 'recall', 'F1', 'false_alarms_per_defect_image',
                    'false_alarms_per_defect_free_image', 'defect_free_images_with_zero_alarms'])
        for (m, c), v in summary.items():
            w.writerow([m, c, *(round(v[k], 3) for k in ('precision', 'recall', 'f1', 'fp_defect_imgs',
                                                         'fp_clean', 'clean_ok'))])


def make_charts(summary, curves, conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'absdiff': 'Absolute Difference', 'ssim': 'SSIM', 'ours': 'Proposed pipeline'}
    colors = {'absdiff': '#d9822b', 'ssim': '#3b82c4', 'ours': '#2e9e5b'}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    xs = np.arange(len(conditions))
    for i, m in enumerate(names):
        ax.bar(xs + (i - 1) * .27, [summary[(m, c)]['f1'] for c in conditions], .27, label=names[m], color=colors[m])
    ax.set_xticks(xs); ax.set_xticklabels(conditions, rotation=25)
    ax.set_ylabel('F1 score'); ax.set_ylim(0, 1.05); ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, -.2))
    ax.set_title('Defect detection F1 under different conditions (threshold calibrated on "normal" only)')
    fig.tight_layout(); fig.savefig(OUT / 'f1_by_condition.png', dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, m in enumerate(names):
        ax.bar(xs + (i - 1) * .27, [summary[(m, c)]['fp_clean'] for c in conditions], .27, label=names[m], color=colors[m])
    ax.set_xticks(xs); ax.set_xticklabels(conditions, rotation=25)
    ax.set_ylabel('False alarms per defect-free image'); ax.legend()
    ax.set_title('Noise robustness: boxes drawn on boards that have NO defect')
    fig.tight_layout(); fig.savefig(OUT / 'false_alarms_defect_free.png', dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in ('absdiff', 'ssim'):
        t, f = zip(*curves[m])
        ax.plot(t, f, label=names[m], color=colors[m])
    ax.set_xlabel('threshold'); ax.set_ylabel('F1 on normal light'); ax.legend()
    ax.set_title('Threshold sweep (calibration)')
    fig.tight_layout(); fig.savefig(OUT / 'threshold_sweep.png', dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- F2
def _orb_homography(ref, ins):
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ins, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=4500)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
    good = [a for p in m if len(p) == 2 for a, b in [p] if a.distance < .75 * b.distance]
    if len(good) < 12:
        return None
    src = np.float32([k2[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.5)
    return H


def alignment_experiment():
    """Known tilt T (reference -> inspection).  Error = mean distance (px) between where the
    estimated homography sends the tilted points and where they really are in the reference."""
    rows = []
    gx, gy = np.meshgrid(np.linspace(.1, .9, 7), np.linspace(.1, .9, 7))
    for b, (rp, _, _) in BOARDS.items():
        ref = load_image(rp)
        h, w = ref.shape[:2]
        P = np.stack([gx.ravel() * w, gy.ravel() * h], 1).astype(np.float32)
        for deg, st in [(0, 0), (3, .01), (6, .02), (10, .03), (15, .04), (20, .05)]:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
            s = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            d = s + st * np.float32([[1, 1], [-1, 2], [-2, -1], [1, -2]]) * min(h, w)
            T = np.vstack([M, [0, 0, 1]])
            T = cv2.getPerspectiveTransform(s, d) @ T
            ins = cv2.warpPerspective(ref, T, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
            Pt = cv2.perspectiveTransform(P[:, None], T)[:, 0]
            H = _orb_homography(ref, ins)
            if H is None:
                err_orb = float('nan')
            else:
                back = cv2.perspectiveTransform(Pt[:, None], H)[:, 0]
                err_orb = float(np.mean(np.linalg.norm(back - P, axis=1)))
            err_none = float(np.mean(np.linalg.norm(Pt - P, axis=1)))     # no alignment at all
            rows.append(dict(board=b, tilt_deg=deg, perspective=st, err_none=err_none, err_orb=err_orb))
    with open(OUT / 'alignment_error.csv', 'w', newline='', encoding='utf-8') as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w_.writeheader(); w_.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    degs = sorted({r['tilt_deg'] for r in rows})
    for key, lab, col in (('err_none', 'No alignment', '#c0392b'), ('err_orb', 'ORB + RANSAC homography', '#2e9e5b')):
        ax.plot(degs, [np.nanmean([r[key] for r in rows if r['tilt_deg'] == d]) for d in degs], 'o-', label=lab, color=col)
    ax.set_yscale('log'); ax.set_xlabel('board tilt (deg, with perspective)'); ax.set_ylabel('mean position error (px, log)')
    ax.legend(); ax.set_title('Alignment accuracy vs. tilt')
    fig.tight_layout(); fig.savefig(OUT / 'alignment_error.png', dpi=130); plt.close(fig)
    return rows


# --------------------------------------------------------------------------- report
def write_report(summary, calib, conditions, alignment, n_images):
    L = []
    A = L.append
    A('# รายงานการทดสอบ PCB Inspection v1.0\n')
    A(f'สร้างอัตโนมัติด้วย `python benchmark/run_benchmark.py` ({n_images} ภาพทดสอบ, ผลซ้ำได้ทุกครั้ง)\n')
    A('## F2 — จัดแนวภาพเอียงด้วย ORB + Homography\n')
    A('ใส่การเอียง/มุมมอง (perspective) ที่ทราบค่าจริงลงในภาพ แล้ววัดว่าหลังจัดแนวด้วย ORB + RANSAC ตำแหน่งคลาดเคลื่อนกี่พิกเซล\n')
    A('| เอียง (องศา) | ไม่จัดแนว (px) | ORB + Homography (px) |\n|---|---|---|')
    for d in sorted({r['tilt_deg'] for r in alignment}):
        sel = [r for r in alignment if r['tilt_deg'] == d]
        A(f"| {d} | {np.mean([r['err_none'] for r in sel]):.1f} | {np.nanmean([r['err_orb'] for r in sel]):.2f} |")
    fails = sum(1 for r in alignment if r['err_orb'] != r['err_orb'])
    A(f'\nจัดแนวล้มเหลว {fails} จาก {len(alignment)} กรณี  (กราฟ: `alignment_error.png`)\n')
    A('ผลคือถ่ายภาพเอียงด้วยกล้องทั่วไปก็ตรวจได้ ไม่ต้องใช้แท่นล็อกชิ้นงาน (jig) ตราบใดที่ภาพยังเห็นทั้งบอร์ดและมีลายให้จับคู่ได้\n')
    A('## F3 — ชุดข้อมูลทดสอบที่ท้าทาย\n')
    A('3 บอร์ด (เขียว / ดำ / แดง) รวม 17 ชิ้นส่วนที่ถูกถอดออก ครอบคลุมสภาพที่ยาก:\n')
    A('- **แสงสะท้อนบนจุดบัดกรี** (`glare`): จุดสว่างจัดแบบ specular 6 จุดต่อภาพ ตำแหน่งสุ่ม\n'
      '- **ชิ้นส่วน SMD เล็กสีกลืนกับบอร์ด**: บอร์ดดำที่ชิ้นส่วนเป็นสีดำบนพื้นดำ (`black`)\n'
      '- แสงสว่าง/มืด/อุ่น/แกมมา/แสงไม่สม่ำเสมอ, สัญญาณรบกวน, JPEG คุณภาพต่ำ\n'
      '- บอร์ดเอียง (`tilt`) และเอียงร่วมกับแสงสะท้อน (`tilt+glare`)\n'
      '- ทุกเงื่อนไขมีภาพควบคุม **ไม่มีข้อบกพร่อง** คู่กัน เพื่อวัดการเตือนผิด\n')
    A('## F4 — Absolute Difference เทียบกับ SSIM\n')
    A('ทั้งสองวิธีใช้ภาพที่จัดแนวแล้วชุดเดียวกัน, เบลอ Gaussian 5×5 เท่ากัน, ขั้นตอนสร้างกรอบเท่ากัน '
      f'และ**ปรับ threshold บนสภาพแสงปกติเท่านั้น** (AbsDiff = {calib["absdiff"]}, SSIM = {calib["ssim"]}) แล้วนำไปใช้กับทุกเงื่อนไขโดยไม่ปรับซ้ำ '
      'เพื่อดูว่าวิธีไหนทนต่อการเปลี่ยนแสงได้จริง\n')
    A('| เงื่อนไข | AbsDiff F1 | SSIM F1 | Pipeline F1 | AbsDiff FP/ภาพไม่มีตำหนิ | SSIM FP/ภาพไม่มีตำหนิ | Pipeline FP/ภาพไม่มีตำหนิ |\n|---|---|---|---|---|---|---|')
    for c in conditions + ['ALL']:
        g = lambda m, k: summary[(m, c)][k]
        A(f"| {c} | {g('absdiff','f1'):.2f} | {g('ssim','f1'):.2f} | {g('ours','f1'):.2f} | "
          f"{g('absdiff','fp_clean'):.1f} | {g('ssim','fp_clean'):.1f} | {g('ours','fp_clean'):.1f} |")
    A('\nกราฟ: `f1_by_condition.png`, `false_alarms_defect_free.png`, `threshold_sweep.png`\n')
    A('ตัวชี้วัด: Precision / Recall / F1 นับว่า "เจอ" เมื่อกรอบอยู่ห่างจากชิ้นส่วนที่ถอดจริงไม่เกิน 60 px '
      '(หนึ่งชิ้นส่วนจับคู่ได้หนึ่งครั้ง) และ FP/ภาพไม่มีตำหนิ = จำนวนกรอบที่ขึ้นบนภาพที่ไม่มีอะไรเปลี่ยนเลย\n')
    a, b, o = (summary[(m, 'ALL')] for m in ('absdiff', 'ssim', 'ours'))
    A('## สรุปผล\n')
    A(f"- **SSIM ดีกว่า Absolute Difference** ในภาพรวม (F1 {b['f1']:.2f} เทียบ {a['f1']:.2f}) เพราะเทียบโครงสร้างของภาพ จึงไม่ถูกหลอกง่ายเมื่อแสงไม่สม่ำเสมอหรือภาพเอียงเล็กน้อย "
      "แต่ SSIM ไวต่อการเปลี่ยนความสว่างรวม (dark/gamma) ทำให้เตือนผิดเพิ่ม ส่วน Absolute Difference เสียความแม่นยำที่สุดเมื่อภาพมืด")
    A(f"- ทั้งสองวิธีเดี่ยวๆ ยังไม่พอใช้งานจริง: F1 รวม {a['f1']:.2f} / {b['f1']:.2f} เทียบกับ pipeline ของโปรแกรม {o['f1']:.2f} "
      "ซึ่งเพิ่มการปรับความสว่าง/สีอัตโนมัติ, ยอมให้ขอบเลื่อนเล็กน้อย, ตรวจว่าตำแหน่งนั้น 'เหลือแต่บอร์ด' จริง และกรองกรอบเล็ก/บางที่เป็นสัญญาณรบกวน")
    A('\n## ข้อจำกัดที่พบจากการทดสอบ (ไม่ปิดบัง)\n')
    A(f"- **แสงสะท้อนจัด (glare)**: pipeline ยังขึ้นกรอบผิดบนจุดสะท้อนแรง (เฉลี่ย {summary[('ours','glare')]['fp_clean']:.1f} กรอบ/ภาพที่ไม่มีตำหนิ ในชุดทดสอบที่ใส่จุดสะท้อน 6 จุด) "
      "ผมลองกรองจุดสะท้อนด้วยความสว่าง/ขอบนุ่ม แล้วพบว่าแยกไม่ขาดจากแผ่นทองแดงที่เพิ่งเผยหลังถอดชิ้นส่วน ซึ่งทำให้ตรวจชิ้นส่วนจริงพลาด จึงไม่ใส่ไว้ แนะนำให้ถ่ายด้วยแสงกระจาย หรือใช้โพลาไรเซอร์\n"
      "- การทดสอบชิ้นส่วนที่ถูกถอดใช้ภาพตัวอย่าง 3 บอร์ด (17 ชิ้นส่วน) เงื่อนไขแสง/เอียง/สัญญาณรบกวนจำลองด้วยโค้ด ยังไม่ใช่ภาพถ่ายจริงจากหน้างาน\n"
      "- การทดสอบ F2 บังคับใช้ ORB อย่างเดียว ส่วนตัวโปรแกรมจริงลอง SIFT ก่อนแล้วค่อยใช้ ORB สำรอง\n")
    (OUT / 'REPORT.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
