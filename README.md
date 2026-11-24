# PCB Inspection v1.0 — rewritten, one detector for every board

Run: double-click `run_windows.bat` → choose Reference + Inspection → **Inspect PCB**.

What changed vs 0.9.x
- Removed the whole stack of per-colour routes, YOLO model, and the R/C/Diode/LED classification.
  It only answers: **what part that was on the reference is gone in the inspection photo?**
- No `ultralytics` / model download. Starts fast, installs only opencv + numpy + Pillow.
- RED box = part gone (bare board now). YELLOW box = area changed, please look.
- Many boards: `python batch.py REFERENCE.png FOLDER` → marked image per board + summary.csv.

How it works (simple_detect.py): align → fit lighting/colour → shift-tolerant colour difference →
split merged blobs with rising thresholds → keep only solid, strong, structurally-different blobs.

Tests: `python -m unittest discover -s tests`

## Benchmark (F2 / F3 / F4)

`python benchmark/run_benchmark.py` (needs `pip install scikit-image matplotlib`) rebuilds everything in
`benchmark/results/`: `REPORT.md` (Thai summary), CSV tables and charts for ORB+homography alignment accuracy,
the challenging test set (glare, uneven light, noise, JPEG, tilt) and Absolute Difference vs SSIM vs this pipeline.
