import unittest
from pathlib import Path

from emotion import DifficultyPolicy, ExpressionObservation, ExpressionRecognizer


def observation(negative_mass, confidence=None):
    probabilities = {"sad": negative_mass, "neutral": 1 - negative_mass}
    label = max(probabilities, key=probabilities.get)
    return ExpressionObservation(
        label,
        probabilities[label] if confidence is None else confidence,
        probabilities,
        0.95,
    )


class DifficultyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = DifficultyPolicy()

    def feed(self, start, masses):
        for index, mass in enumerate(masses):
            self.policy.observe(observation(mass), start + index * 0.25)

    def test_sustained_evidence_lowers_once_and_observes_cooldown(self):
        for turn, expected in enumerate(("lowered", "cooldown", "cooldown", "lowered", "minimum")):
            start = turn * 2
            self.feed(start, [0.9] * 5)
            decision = self.policy.decide(start + 1.25)
            self.assertEqual(decision.reason, expected)
        self.assertEqual(self.policy.difficulty, 1)
        self.policy.reset()
        self.assertEqual(self.policy.difficulty, 3)

    def test_weighted_mean_and_agreement_must_both_pass(self):
        self.feed(0, [0.99, 0.99, 0.99, 0.64, 0.64])
        agreement_failure = self.policy.decide(1.25)
        self.assertGreater(agreement_failure.negative_score, 0.70)
        self.assertEqual(agreement_failure.reason, "unchanged")

        self.feed(2, [0.72, 0.72, 0.72, 0.72, 0.10])
        mean_failure = self.policy.decide(3.25)
        self.assertLess(mean_failure.negative_score, 0.70)
        self.assertEqual(mean_failure.reason, "unchanged")
        self.assertEqual(self.policy.difficulty, 3)

    def test_missing_stale_and_low_confidence_evidence_cannot_lower_difficulty(self):
        self.feed(0, [0.9] * 5)
        self.assertEqual(self.policy.decide(3).reason, "insufficient_evidence")

        self.feed(4, [0.9] * 5)
        self.policy.observe(None, 5.1)
        self.assertEqual(self.policy.decide(5.25).sample_count, 0)

        for index in range(5):
            self.policy.observe(observation(0.9, confidence=0.54), 6 + index * 0.25)
        self.assertEqual(self.policy.decide(7.25).sample_count, 0)
        self.assertEqual(self.policy.difficulty, 3)

    def test_samples_are_consumed_at_each_decision_and_cleared_on_phase_change(self):
        self.feed(0, [0.9] * 3)
        self.policy.clear_evidence()
        self.feed(1, [0.9] * 3)
        self.assertEqual(self.policy.decide(1.6).sample_count, 3)
        self.assertEqual(self.policy.decide(1.7).sample_count, 0)


class ExpressionRecognizerTests(unittest.TestCase):
    def test_missing_models_disable_inference(self):
        recognizer = ExpressionRecognizer(Path(__file__).with_suffix("") / "models")
        self.assertFalse(recognizer.available)
        self.assertIsNotNone(recognizer.error)
        self.assertIsNone(recognizer.recognize(None))

    def test_bundled_models_load_and_blank_frame_returns_no_expression(self):
        import numpy as np

        recognizer = ExpressionRecognizer()
        self.assertTrue(recognizer.available, recognizer.error)
        self.assertIsNone(recognizer.recognize(np.zeros((240, 320, 3), dtype=np.uint8)))
        self.assertIsNone(recognizer.error)
