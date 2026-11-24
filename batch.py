"""Check many boards in one go.

  python batch.py REFERENCE.png  FOLDER_WITH_INSPECTION_IMAGES  [OUT_FOLDER]

Writes <name>_result.png (marked image) per inspection image + summary.csv.
"""
import csv
import sys
from pathlib import Path
from pcb_inspector.core import inspect, load_image, write_image

def main():
    if len(sys.argv) < 3:
        print(__doc__); return 1
    ref = load_image(sys.argv[1])
    folder = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else folder / 'results'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') or p.resolve() == Path(sys.argv[1]).resolve():
            continue
        try:
            r = inspect(ref, load_image(p))
            write_image(out / (p.stem + '_result.png'), r.overlay)
            miss = sum(f.review_status == 'Likely missing' for f in r.findings)
            rows.append([p.name, miss, len(r.findings) - miss, r.registration])
            print(f'{p.name}: {miss} missing, {len(r.findings) - miss} changed')
        except Exception as e:
            rows.append([p.name, 'ERROR', '', str(e)])
            print(f'{p.name}: ERROR {e}')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['file', 'missing', 'changed', 'registration']); w.writerows(rows)
    return 0

if __name__ == '__main__':
    sys.exit(main())
