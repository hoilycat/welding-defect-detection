from __future__ import annotations

import unittest

from rules import explain_detection


class RulesTest(unittest.TestCase):
    def test_lack_of_fusion_uses_distinct_korean_name(self) -> None:
        result = explain_detection("lack_of_fusion")

        self.assertEqual(result.display_name, "융합불량 (Lack of Fusion)")

    def test_incomplete_penetration_is_labeled_as_penetration_defect(self) -> None:
        result = explain_detection("incomplete_penetration")

        self.assertEqual(result.display_name, "용입부족 (Incomplete Penetration)")


if __name__ == "__main__":
    unittest.main()
