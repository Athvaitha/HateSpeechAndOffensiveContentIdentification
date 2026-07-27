import unittest

from predict import get_meme_meaning_and_reason


class PredictExplanationTest(unittest.TestCase):
    def test_explanation_mentions_grounding_evidence(self):
        caption, reason = get_meme_meaning_and_reason("image_telugu_0031.png", 0)

        self.assertTrue(caption.strip(), "expected a caption from dataset metadata")
        self.assertIn("ocr", reason.lower(), "reason should mention OCR grounding for the meme")
        self.assertTrue(
            "caption" in reason.lower() or "image" in reason.lower(),
            "reason should tie the label to image/caption evidence",
        )


if __name__ == "__main__":
    unittest.main()
