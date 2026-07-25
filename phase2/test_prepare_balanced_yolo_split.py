from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prepare_balanced_yolo_split import select_balanced_images


class PrepareBalancedYoloSplitTest(unittest.TestCase):
    def test_keeps_minority_and_limits_dominant_and_normal_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "images" / "train"
            label_root = root / "labels" / "train"
            image_root.mkdir(parents=True)
            label_root.mkdir(parents=True)

            samples = {
                "crack": "0 0.5 0.5 0.2 0.2\n",
                "slag": "3 0.5 0.5 0.2 0.2\n",
                "porosity_a": "1 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n",
                "porosity_b": "1 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n",
                "normal_a": "",
                "normal_b": "",
            }
            for stem, label in samples.items():
                (image_root / f"{stem}.jpg").write_bytes(b"image")
                (label_root / f"{stem}.txt").write_text(label, encoding="utf-8")

            selected, counts, stats = select_balanced_images(
                dataset_root=root,
                dominant_class_id=1,
                target_dominant_boxes=2,
                max_normal_images=1,
                seed=42,
            )

            stems = {path.stem for path in selected}
            self.assertIn("crack", stems)
            self.assertIn("slag", stems)
            self.assertEqual(len(stems & {"porosity_a", "porosity_b"}), 1)
            self.assertEqual(len(stems & {"normal_a", "normal_b"}), 1)
            self.assertEqual(counts[1], 2)
            self.assertEqual(stats["total_images_selected"], 4)


if __name__ == "__main__":
    unittest.main()
