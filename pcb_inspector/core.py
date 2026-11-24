"""PCB inspection core (v1.0): align two photos, report parts that disappeared.

No component classification and no AI model: one generic detector in
simple_detect.py works for green / black / red / blue boards.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json

import cv2
import numpy as np

from .finding_overlay import render_findings
from .simple_detect import detect
from .structure import structure_boxes


@dataclass
class Finding:
    x: int
    y: int
    w: int
    h: int
    score: float
    review_status: str = "Needs review"      # 'Likely missing' | 'Needs review' | 'Low-priority change'
    evidence_reason: str = ""
    component_type: str = "Missing part"
    missing_likelihood: str = ""
    user_decision: str = ""
    occlusion_detected: bool = False
    display_box: tuple | None = None

    def as_dict(self):
        return {"x": self.x, "y": self.y, "width": self.w, "height": self.h,
                "status": self.review_status, "score": round(self.score, 3),
                "reason": self.evidence_reason, "user_decision": self.user_decision}


@dataclass
class InspectionResult:
    reference: np.ndarray
    aligned: np.ndarray
    overlay: np.ndarray
    difference: np.ndarray
    mask: np.ndarray
    findings: list
    registration: str
    valid_coverage: float
    review_findings: list | None = None
    quality_warnings: list | None = None
    ai_diagnostics: dict | None = None
    ai_reference_overlay: np.ndarray | None = None


def load_image(path: str | Path) -> np.ndarray:
    """Support Unicode file paths on Windows."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot open image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    extension = Path(path).suffix.lower() or '.png'
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {extension}")
    encoded.tofile(str(path))


def _photometric_match(reference: np.ndarray, inspected: np.ndarray):
    """Check whether image pixels share coordinates despite uniform light changes.

    Returns the illumination-normalized inspection image only when the vast
    majority of pixels agree after per-channel median-offset compensation.
    This prevents rewarping tiny SMD parts in edited/camera-fixed image pairs.
    """
    if reference.shape != inspected.shape:
        return None
    # Keep medians robust when a small number of components were removed.
    delta = reference.astype(np.int16) - inspected.astype(np.int16)
    offsets = np.median(delta.reshape(-1, 3), axis=0).round().astype(np.int16)
    normalized = np.clip(inspected.astype(np.int16) + offsets, 0, 255).astype(np.uint8)
    # Saturated highlights/shadows have irreversibly lost intensity detail.
    # Treat them as ambiguous only when the reference is near that saturation
    # level too; do NOT suppress a changed component against a dark reference.
    for channel in range(3):
        offset = int(offsets[channel])
        if offset < -3:
            clipped = ((inspected[:, :, channel] == 255) &
                       (reference[:, :, channel] >= 255 + offset - 2))
            normalized[:, :, channel][clipped] = reference[:, :, channel][clipped]
        elif offset > 3:
            clipped = ((inspected[:, :, channel] == 0) &
                       (reference[:, :, channel] <= offset + 2))
            normalized[:, :, channel][clipped] = reference[:, :, channel][clipped]
    residual = np.max(cv2.absdiff(reference, normalized), axis=2)
    # Cap the allowed global illumination correction: unrelated images should
    # not become 'pixel aligned' via an arbitrary large color translation.
    if np.max(np.abs(offsets)) > 35:
        return None
    agreement = float(np.mean(residual <= 4))
    if agreement < .88:
        return None
    return normalized, agreement, offsets


def align_images(reference: np.ndarray, inspected: np.ndarray, feature: str = 'auto'):
    """Align inspected to reference, using SIFT/ORB + RANSAC; ECC fallback.

    ECC-only fallback is for pre-cropped, almost aligned images and is capped to
    a small affine motion so an unrelated photo is not silently accepted.
    """
    h, w = reference.shape[:2]
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(inspected, cv2.COLOR_BGR2GRAY)
    # Image edits and fixed-camera captures often share exact geometry even if
    # illumination changed. Avoid interpolation artifacts on tiny SMD parts.
    photometric = _photometric_match(reference, inspected) if feature == 'auto' else None
    if photometric is not None:
        _, agreement, offsets = photometric
        if np.max(np.abs(offsets)) <= 3:
            return inspected.copy(), np.ones((h, w), bool), f"Pixel-perfect registration ({agreement:.1%} unchanged)"
        return inspected.copy(), np.ones((h, w), bool), f"Photometric-aligned registration ({agreement:.1%} matched)"
    # Homography is in original inspected-image pixels -> reference pixels.
    for extractor in ([cv2.SIFT_create(nfeatures=3500)] if hasattr(cv2, 'SIFT_create') and feature == 'auto' else []) + [cv2.ORB_create(nfeatures=4500)]:
        k1, d1 = extractor.detectAndCompute(ref_gray, None)
        k2, d2 = extractor.detectAndCompute(test_gray, None)
        if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
            continue
        norm = cv2.NORM_L2 if d1.dtype == np.float32 else cv2.NORM_HAMMING
        matches = cv2.BFMatcher(norm).knnMatch(d2, d1, k=2)
        good = [a for pair in matches if len(pair) == 2 for a, b in [pair] if a.distance < 0.72 * b.distance]
        if len(good) < 12:
            continue
        src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.5)
        if H is None or inlier_mask is None or np.count_nonzero(inlier_mask) < 10:
            continue
        inliers = int(np.count_nonzero(inlier_mask))
        if inliers / len(good) < 0.35:
            continue
        corners = np.float32([[[0,0]], [[inspected.shape[1],0]], [[inspected.shape[1], inspected.shape[0]]], [[0, inspected.shape[0]]]])
        transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        poly_area = abs(cv2.contourArea(transformed.astype(np.float32)))
        if not (0.3 * w * h <= poly_area <= 3 * w * h) or not cv2.isContourConvex(transformed.astype(np.float32)):
            continue
        aligned = cv2.warpPerspective(inspected, H, (w, h))
        valid = cv2.warpPerspective(np.full(inspected.shape[:2], 255, np.uint8), H, (w, h)) > 0
        name = 'ORB' if 'ORB' in type(extractor).__name__ else 'SIFT'
        return aligned, valid, f"{name} feature matching + RANSAC homography ({inliers} inliers)"

    # For same-view / cropped photos: resize to reference then perform small affine ECC.
    resized = cv2.resize(inspected, (w, h), interpolation=cv2.INTER_AREA)
    gray2 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ref_norm = cv2.GaussianBlur(ref_gray, (5, 5), 0).astype(np.float32) / 255
    test_norm = cv2.GaussianBlur(gray2, (5, 5), 0).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(ref_norm, test_norm, warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 150, 1e-6),
                                       None, 5)
        angle = np.arctan2(warp[1,0], warp[0,0])
        if cc < 0.55 or abs(angle) > np.deg2rad(12) or abs(warp[0,2]) > w*.12 or abs(warp[1,2]) > h*.12:
            raise ValueError('Images do not appear to be the same board/view')
        aligned = cv2.warpAffine(resized, warp, (w,h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        valid = cv2.warpAffine(np.full((h,w),255,np.uint8), warp, (w,h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        return aligned, valid, f"ECC registration (correlation {cc:.2f})"
    except cv2.error as exc:
        raise ValueError('Image alignment failed. Use clear, top-down photos of the same board with similar framing.') from exc


def refine_alignment(reference, aligned, valid):
    """Remove the smooth leftover misalignment of a homography (lens/perspective,
    different renders). The flow is blurred heavily so a removed part cannot be
    'followed' by it - only board-scale warping is corrected."""
    h, w = reference.shape[:2]
    scale = min(1.0, 700 / max(h, w))
    small = (round(w * scale), round(h * scale))
    g1 = cv2.cvtColor(cv2.resize(reference, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(aligned, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (0, 0), 1.2)
    g2 = cv2.GaussianBlur(g2, (0, 0), 1.2)
    # contrast-normalise so different renders/lighting do not bias the flow
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    g1, g2 = clahe.apply(g1), clahe.apply(g2)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(g1, g2, None)
    sigma = max(6.0, .035 * min(small))
    flow = cv2.GaussianBlur(flow, (0, 0), sigma)
    # never trust large corrections
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    if float(np.percentile(mag, 95)) > .06 * min(small):
        return aligned
    flow_full = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(aligned, gx + flow_full[:, :, 0], gy + flow_full[:, :, 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def image_quality_warnings(reference: np.ndarray, inspected: np.ndarray) -> list[str]:
    """Non-blocking capture advice; deliberately not a defect confidence score."""
    notices = []
    for label, image in (("Reference", reference), ("Inspection", inspected)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if min(image.shape[:2]) < 220:
            notices.append(f"{label} image is low resolution; small SMD parts may be missed")
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 28:
            notices.append(f"{label} image may be blurred; use a steady, focused top-down shot")
        shadow = float(np.mean(gray < 13))
        bright = float(np.mean(gray > 248))
        if shadow > .65 or bright > .65:
            notices.append(f"{label} image contains large saturated dark/bright regions")
    if abs(reference.shape[1]/reference.shape[0] - inspected.shape[1]/inspected.shape[0]) > .16:
        notices.append("Image aspect ratios differ; verify board alignment carefully")
    return notices


def crop_to_board(image: np.ndarray, margin: float = .012):
    """Remove flat padding (white / black / any single-colour frame, repeatedly)
    so that both pictures are compared on the board itself, not on its canvas."""
    out = image
    last = None
    ox = oy = 0
    for _ in range(4):
        h, w = out.shape[:2]
        if min(h, w) < 120:
            break
        t = max(2, int(min(h, w) * .01))
        edge = np.concatenate([out[:t].reshape(-1, 3), out[-t:].reshape(-1, 3),
                               out[:, :t].reshape(-1, 3), out[:, -t:].reshape(-1, 3)])
        if np.mean(np.std(edge.astype(np.float32), axis=0)) > 14:
            break                      # border is not a flat colour: nothing to remove
        bg = np.median(edge, axis=0).astype(np.float32)
        diff = np.abs(out.astype(np.float32) - bg).max(axis=2)
        fg = (diff > 28).astype(np.uint8)
        k = max(3, int(min(h, w) * .006)) | 1
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        cols = np.where(fg.mean(axis=0) > .01)[0]
        rows = np.where(fg.mean(axis=1) > .01)[0]
        if len(cols) < 20 or len(rows) < 20:
            break
        x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        if (x1 - x0) * (y1 - y0) > .93 * h * w:
            break
        last = (out, x0, x1, y0, y1, ox, oy)
        out = out[y0:y1, x0:x1]
        ox += x0
        oy += y0
    if last is not None:
        src, x0, x1, y0, y1, ox, oy = last
        h, w = src.shape[:2]
        mx, my = int(margin * (x1 - x0)), int(margin * (y1 - y0))
        x0, y0 = max(0, x0 - mx), max(0, y0 - my)
        out = src[y0:min(h, y1 + my), x0:min(w, x1 + mx)]
        ox += x0
        oy += y0
    return np.ascontiguousarray(out), (ox, oy)


def _small_gone(b, reference) -> bool:
    """A small box that is clearly bare board now (strong, 'gone'): a similar part next
    door must not talk us out of it."""
    return (b.status == 'Likely missing' and b.strength / 4.0 >= .5
            and max(b.w, b.h) <= .12 * min(reference.shape[:2]))


def _part_still_there(reference, aligned, b) -> bool:
    """True when what the reference shows inside the box can still be found in the
    inspection picture a little way off: the part exists, it is merely drawn / placed
    slightly differently (different render or photo), so it is not a defect."""
    gr = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ga = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    H, W = gr.shape
    x0, y0, x1, y1 = int(b.x), int(b.y), int(b.x + b.w), int(b.y + b.h)
    patch = gr[y0:y1, x0:x1]
    if patch.size < 100:
        return False
    if patch.std() < 8:
        # plain, featureless part (a pad, a flat can): same colour in the inspection
        # picture means it is still there
        lab_r = cv2.cvtColor(reference[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
        lab_a = cv2.cvtColor(aligned[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
        return float(np.linalg.norm(lab_r - lab_a)) < 14
    m = int(.4 * min(b.w, b.h)) + 5
    win = ga[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)]
    if win.shape[0] < patch.shape[0] or win.shape[1] < patch.shape[1]:
        return False
    res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(res)
    if best <= .40:
        return False
    # the match must lie close to where the part was; a similar-looking neighbour
    # (another capacitor a few pixels away) does not mean this one is still there
    off_x = loc[0] - (x0 - max(0, x0 - m))
    off_y = loc[1] - (y0 - max(0, y0 - m))
    if max(b.w, b.h) > .08 * min(H, W):
        return True               # big parts: any nearby match is enough (renders differ)
    return max(abs(off_x), abs(off_y)) <= max(4, .3 * min(b.w, b.h))


NO_MERGE = False


def _is_printed_text(reference, aligned, f, short) -> bool:
    """Silkscreen lettering exists in both pictures (only drawn a little differently):
    both patches hold a similar share of thin white strokes.  A part that is gone has
    white strokes (or a white body) in the reference and none in the inspection."""
    if f.review_status != 'Likely missing' or max(f.w, f.h) > .04 * short:
        return False              # lettering is small; a big box is a real part
    def white(im):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        return float(np.mean((hsv[..., 2] > 190) & (hsv[..., 1] < 70)))
    a = white(reference[f.y:f.y + f.h, f.x:f.x + f.w])
    b = white(aligned[f.y:f.y + f.h, f.x:f.x + f.w])
    return min(a, b) >= .15 and min(a, b) / max(a, b) >= .55


def _drop_duplicates(findings):
    """Two boxes on the same spot are one finding: keep the surer one (red before yellow)."""
    order = sorted(findings, key=lambda f: (f.review_status != 'Likely missing', -f.score))
    keep = []
    for f in order:
        dup = False
        for g in keep:
            iw = max(0, min(f.x + f.w, g.x + g.w) - max(f.x, g.x))
            ih = max(0, min(f.y + f.h, g.y + g.h) - max(f.y, g.y))
            if iw * ih > .25 * min(f.w * f.h, g.w * g.h):
                dup = True
                break
        if not dup:
            keep.append(f)
    return [f for f in findings if any(f is k for k in keep)]


def _merge_split_parts(reference, findings):
    """One big chip that is gone shows up as several fragments (its dark body matches the
    dark board in places).  When fragments sit on ONE connected patch of uniform reference
    colour, they are one part: report it once, boxed on that patch."""
    red = [f for f in findings if f.review_status == 'Likely missing' and f.component_type == 'Missing part']
    if len(red) < 2:
        return findings
    H, W = reference.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(reference, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    used, out = set(), []
    for i, f in enumerate(red):
        if i in used:
            continue
        cx, cy = int(f.x + f.w / 2), int(f.y + f.h / 2)
        r = max(2, min(f.w, f.h) // 5)
        seed = lab[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].reshape(-1, 3).mean(0)
        same = (np.linalg.norm(lab - seed, axis=2) < 16).astype(np.uint8)
        same = cv2.morphologyEx(same, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, comp = cv2.connectedComponents(same, connectivity=4)
        cid = comp[min(H - 1, cy), min(W - 1, cx)]
        if cid == 0:
            continue
        mine = comp == cid
        group = [j for j, g in enumerate(red) if j not in used and
                 mine[min(H - 1, int(g.y + g.h / 2)), min(W - 1, int(g.x + g.w / 2))]]
        if len(group) < 2:
            continue
        ux0 = min(red[j].x for j in group); uy0 = min(red[j].y for j in group)
        ux1 = max(red[j].x + red[j].w for j in group); uy1 = max(red[j].y + red[j].h for j in group)
        ys, xs = np.where(mine[max(0, uy0 - (uy1 - uy0) // 4):min(H, uy1 + (uy1 - uy0) // 4),
                               max(0, ux0 - (ux1 - ux0) // 4):min(W, ux1 + (ux1 - ux0) // 4)])
        if len(xs) == 0:
            continue
        ox, oy = max(0, ux0 - (ux1 - ux0) // 4), max(0, uy0 - (uy1 - uy0) // 4)
        bx0, bx1, by0, by1 = ox + xs.min(), ox + xs.max() + 1, oy + ys.min(), oy + ys.max() + 1
        gx, gy = max(3, int(.08 * (ux1 - ux0))), max(3, int(.08 * (uy1 - uy0)))
        bx0, bx1 = max(bx0, ux0 - gx), min(bx1, ux1 + gx)       # stay near the fragments:
        by0, by1 = max(by0, uy0 - gy), min(by1, uy1 + gy)       # the patch may leak a little
        area_u = (ux1 - ux0) * (uy1 - uy0)
        if int(mine.sum()) > 2.5 * area_u:
            continue          # that colour is the board itself, not one part
        if (bx1 - bx0) * (by1 - by0) > 2.2 * area_u or (bx1 - bx0) * (by1 - by0) > .12 * H * W:
            continue          # the colour patch leaks into something else: leave the fragments alone
        used.update(group)
        best = max((red[j] for j in group), key=lambda g: g.score)
        out.append(Finding(int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0), best.score, 'Likely missing',
                           best.evidence_reason, component_type='Missing part', missing_likelihood='Part is gone'))
    if not used:
        return findings
    keep = [f for f in findings if not any(f is red[j] for j in used)]
    return out + keep


def inspect(reference: np.ndarray, inspected: np.ndarray, model_dir=None, pretrained_detections=None) -> InspectionResult:
    """Align `inspected` to `reference` and list parts that are gone / changed."""
    if reference is None or inspected is None or reference.ndim != 3 or inspected.ndim != 3:
        raise ValueError("Select two valid color PCB images")
    full_reference = reference
    orig_rh, orig_rw = reference.shape[:2]
    orig_ih, orig_iw = inspected.shape[:2]
    reference, (ox, oy) = crop_to_board(reference)
    inspected, _ = crop_to_board(inspected)
    warnings = image_quality_warnings(reference, inspected)
    aligned, valid, method = align_images(reference, inspected)
    if np.count_nonzero(valid) / valid.size < .65:
        raise ValueError('Less than 65% of the board overlaps. Retake photos with similar framing.')
    # inspection photo of a different size/resolution = a different shot or render:
    # edges will not line up pixel for pixel, so use a wider shift tolerance
    rh, rw = reference.shape[:2]
    ih, iw = inspected.shape[:2]
    wide = abs(orig_rw / orig_iw - 1) > .05 or abs(orig_rh / orig_ih - 1) > .05
    blobs, difference, mask = detect(reference, aligned, valid, wide=wide)
    if wide and not blobs:
        # nothing differs in colour (e.g. black parts on a black board): look at
        # fine structure (pins / markings appearing or disappearing) instead
        blobs, difference, mask = detect(reference, aligned, valid, wide=True, texture=True)
    extra = []
    if wide:
        # different render / photo: colours never agree exactly, so a 'changed colour'
        # alone is not evidence.  Keep only what is really gone, then add places where the
        # part's own structure can no longer be found (dark part -> bare dark board).
        blobs = [b for b in blobs
                 if b.strength / 4.0 >= (.6 if b.status != 'Likely missing' else .3)
                 and (_small_gone(b, reference) or not _part_still_there(reference, aligned, b))]
        for (x, y, w, h) in structure_boxes(reference, aligned, valid):
            area = w * h
            if any(max(0, min(x + w, b.x + b.w) - max(x, b.x)) * max(0, min(y + h, b.y + b.h) - max(y, b.y)) > .25 * min(area, b.w * b.h)
                   for b in blobs):
                continue
            extra.append((x, y, w, h))
    findings = []
    for b in blobs:
        missing = b.status == 'Likely missing'
        findings.append(Finding(
            b.x, b.y, b.w, b.h, min(1.0, b.strength / 4.0), b.status, b.reason,
            component_type='Missing part' if missing else 'Changed area',
            missing_likelihood='Part is gone' if missing else 'Area changed - check'))
    for (x, y, w, h) in extra:
        findings.append(Finding(int(x), int(y), int(w), int(h), .5, 'Likely missing',
                                'the part seen here in the reference can no longer be found',
                                component_type='Missing part', missing_likelihood='Part is gone'))
    if wide:
        findings = [f for f in findings if not _is_printed_text(reference, aligned, f, min(orig_rh, orig_rw))]
    findings = findings if NO_MERGE else _merge_split_parts(reference, findings)
    findings = _drop_duplicates(findings)
    if (ox, oy) != (0, 0) or reference.shape != full_reference.shape:
        # report in the coordinates of the picture the user gave us
        fh, fw = full_reference.shape[:2]
        ch, cw = reference.shape[:2]
        for f in findings:
            f.x = int(f.x + ox)
            f.y = int(f.y + oy)
        canvas = full_reference.copy()
        canvas[oy:oy + ch, ox:ox + cw] = aligned
        aligned = canvas
        for name in ('difference', 'mask'):
            arr = difference if name == 'difference' else mask
            full = np.zeros((fh, fw), arr.dtype)
            full[oy:oy + ch, ox:ox + cw] = arr[:ch, :cw]
            if name == 'difference':
                difference = full
            else:
                mask = full
        reference = full_reference
    overlay = render_findings(aligned, findings)
    return InspectionResult(reference, aligned, overlay, difference, mask, findings, method,
                            float(np.mean(valid)), list(findings), warnings)


def save_report(result: InspectionResult, directory) -> Path:
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    write_image(dest / '01_reference.png', result.reference)
    write_image(dest / '02_aligned_board.png', result.aligned)
    write_image(dest / '03_inspection_overlay.png', result.overlay)
    write_image(dest / '04_difference_heatmap.png', cv2.applyColorMap(result.difference, cv2.COLORMAP_JET))
    write_image(dest / '05_candidate_mask.png', result.mask)
    report = {"registration": result.registration, "overlap_fraction": round(result.valid_coverage, 3),
              "quality_warnings": result.quality_warnings or [],
              "count": len(result.findings),
              "findings": [f.as_dict() for f in result.findings]}
    (dest / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (dest / 'findings.csv').open('w', newline='', encoding='utf-8-sig') as out:
        w = csv.writer(out)
        w.writerow(['id', 'x_px', 'y_px', 'width_px', 'height_px', 'status', 'reason', 'user_decision'])
        for i, f in enumerate(result.findings, 1):
            w.writerow([i, f.x, f.y, f.w, f.h, f.review_status, f.evidence_reason, f.user_decision])
    return dest
