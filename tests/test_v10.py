import unittest
from pathlib import Path
import cv2
import numpy as np
from pcb_inspector.core import inspect, load_image

S = Path(__file__).resolve().parents[1] / 'samples'
PAIRS = {
    'green': (S/'green_reference.png', S/'green_inspection.png',
              [(211, 85), (295, 108), (88, 255), (197, 388), (410, 400), (412, 115)]),
    'black': (S/'black_board'/'reference.png', S/'black_board'/'inspection_aligned.png',
              [(275, 100), (720, 95), (625, 395), (380, 520), (712, 605), (145, 437)]),
    'reka': (S/'reka_reference_regression.png', S/'reka_aligned_regression.png',
             [(136, 366), (125, 440), (365, 613), (627, 280), (660, 455)]),
    'pico': (S/'pico2_reference.png', S/'pico2_missing_components.png', None),
}


def centres(result):
    return [(f.x + f.w / 2, f.y + f.h / 2) for f in result.findings]


class TestDetector(unittest.TestCase):
    def test_every_removed_part_found(self):
        for name, (a, b, gt) in PAIRS.items():
            if gt is None:
                continue
            c = centres(inspect(load_image(a), load_image(b)))
            for gx, gy in gt:
                self.assertTrue(any(abs(x-gx) < 60 and abs(y-gy) < 60 for x, y in c), f'{name}: missed {(gx, gy)}')

    def test_green_board_exactly_six(self):
        a, b, _ = PAIRS['green']
        self.assertEqual(len(inspect(load_image(a), load_image(b)).findings), 6)

    def test_pico_five(self):
        a, b, _ = PAIRS['pico']
        self.assertEqual(len(inspect(load_image(a), load_image(b)).findings), 5)

    def test_identical_and_lighting_changes_give_nothing(self):
        for name, (a, _, _) in PAIRS.items():
            ref = load_image(a)
            self.assertEqual(len(inspect(ref, ref.copy()).findings), 0, name)
            bright = np.clip(ref.astype(np.float32)*1.15+6, 0, 255).astype(np.uint8)
            self.assertEqual(len(inspect(ref, bright).findings), 0, name)
            gamma = (255*((ref/255.0)**1.4)).astype(np.uint8)
            self.assertEqual(len(inspect(ref, gamma).findings), 0, name)

    def test_small_shift_and_rotation_are_not_defects(self):
        ref = load_image(PAIRS['green'][0])
        h, w = ref.shape[:2]
        for M in (np.float32([[1, 0, 5], [0, 1, -4]]), cv2.getRotationMatrix2D((w/2, h/2), 1.5, 1.0)):
            moved = cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            self.assertEqual(len(inspect(ref, moved).findings), 0)


if __name__ == '__main__':
    unittest.main()
