from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prepare_yolo_dataset import CLASS_IDS, _annotation_to_yolo, convert_split


class PrepareYoloDatasetTest(unittest.TestCase):
    def test_polygon_becomes_normalized_bounding_box(self) -> None:
        annotation = {
            "class": "defect",
            "case": "crack",
            "coordinate": {"x": [10, 30, 20], "y": [20, 40, 30]},
        }

        class_name, line = _annotation_to_yolo(annotation, width=100, height=100)

        self.assertEqual(class_name, "crack")
        self.assertEqual(line, f"{CLASS_IDS['crack']} 0.200000 0.300000 0.200000 0.200000")

    def test_normal_annotation_is_ignored(self) -> None:
        annotation = {
            "class": "normal",
            "case": "",
            "coordinate": {"x": [0, 10], "y": [0, 10]},
        }

        self.assertIsNone(_annotation_to_yolo(annotation, width=100, height=100))

    def test_zero_area_polygon_is_rejected(self) -> None:
        annotation = {
            "class": "defect",
            "case": "porosity",
            "coordinate": {"x": [10, 20], "y": [30, 30]},
        }

        with self.assertRaises(ValueError):
            _annotation_to_yolo(annotation, width=100, height=100)

    def test_missing_case_uses_image_fallback(self) -> None:
        annotation = {
            "class": "defect",
            "case": "",
            "coordinate": {"x": [10, 20], "y": [30, 40]},
        }

        class_name, _ = _annotation_to_yolo(
            annotation, width=100, height=100, fallback_case="porosity"
        )

        self.assertEqual(class_name, "porosity")

    def test_split_conversion_writes_image_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "Training" / "01.원천데이터" / "TS_VT_test"
            label_dir = root / "Training" / "02.라벨링데이터" / "TL_VT_test"
            output = root / "output"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / "sample.jpg").write_bytes(b"image")
            document = {
                "info": {"type": "VT"},
                "image_data": {
                    "file_name": "sample",
                    "format": "jpg",
                    "width": 100,
                    "height": 100,
                },
                "annotations": [
                    {
                        "class": "defect",
                        "case": "incomplete penetration",
                        "coordinate": {"x": [20, 40], "y": [30, 50]},
                    }
                ],
            }
            (label_dir / "sample.json").write_text(
                json.dumps(document), encoding="utf-8"
            )

            summary, counts = convert_split(
                source_root=root,
                output_root=output,
                source_split="Training",
                target_split="train",
                inspection_type="VT",
                dry_run=False,
                limit_per_folder=None,
            )

            self.assertEqual(summary.images, 1)
            self.assertEqual(summary.boxes, 1)
            self.assertEqual(counts["incomplete_penetration"], 1)
            self.assertTrue((output / "images" / "train" / "sample.jpg").is_file())
            self.assertTrue((output / "labels" / "train" / "sample.txt").is_file())


if __name__ == "__main__":
    unittest.main()
