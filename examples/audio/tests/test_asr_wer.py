"""Unit test for the shared ASR WER harness (row S69-ENGREV-WHISPERCPP-WERNORM-2026-10-27-01).

Run: python test_asr_wer.py   (no model, no GPU)
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asr_wer  # noqa: E402

PAIRS = [
    ("mr smith paid twenty million dollars on the third", "Mr. Smith paid $20 million on the 3rd"),
    ("the colour of the section twelve", "The color of the Section 12."),
    ("they won't pay one thousand two hundred and fifty", "They will not pay 1,250"),
    ("it was fifty percent", "It was 50%."),
]
# the three tests that used to carry their own copies must import the shared module
CONSUMERS = ["test-asr-tdt-cli.py", "test_audio_roundtrip.py", "test-speak-stream-cli.py"]


class AsrWer(unittest.TestCase):
    def test_pairs_normalise_equal(self):
        for a, b in PAIRS:
            self.assertEqual(asr_wer.normalise(a), asr_wer.normalise(b), (a, b))

    def test_old_normaliser_differs(self):
        # what the hand-rolled copies did: they count every pair as errors
        for a, b in PAIRS:
            self.assertNotEqual(asr_wer.normalise(a, english=False), asr_wer.normalise(b, english=False))

    def test_values(self):
        n = asr_wer.normalise
        self.assertEqual(n("twenty million"), ["20000000"])
        self.assertEqual(n("the twenty third"), ["the", "23rd"])
        self.assertEqual(n("the 11th and the twelfth"), ["the", "11th", "and", "the", "12th"])
        self.assertEqual(n("one hundred twenty one"), ["121"])
        self.assertEqual(n("[music] hello (laughs) world"), ["hello", "world"])

    def test_wer_counts(self):
        self.assertEqual(asr_wer.wer_counts("a b c d", "a x c"), (2, 4))
        self.assertEqual(asr_wer.wer("", ""), 0.0)
        self.assertEqual(asr_wer.wer("", "x"), 1.0)
        self.assertAlmostEqual(asr_wer.wer("paid twenty dollars", "paid $20"), 0.0)

    def test_consumers_use_shared_module(self):
        for name in CONSUMERS:
            with open(os.path.join(HERE, name), encoding="utf-8") as f:
                self.assertIn("import asr_wer", f.read(), name)


if __name__ == "__main__":
    unittest.main()
