"""Local facial expression inference and the per-turn difficulty policy."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path


EXPRESSION_LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
NEGATIVE_LABELS = ("angry", "disgust", "fearful", "sad")
MODEL_DIR = Path(__file__).resolve().parent / "models"
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
EXPRESSION_MODEL = "facial_expression_recognition_mobilefacenet_2022july.onnx"


@dataclass(frozen=True)
class ExpressionObservation:
    label: str
    confidence: float
    probabilities: dict[str, float]
    face_confidence: float


class ExpressionRecognizer:
    """Recognize expressions on one sufficiently large face in a BGR frame."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.available = False
        self.error = None
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            self.error = str(error)
            return

        self._cv = cv2
        self._np = np
        # OpenCV Zoo's five-point 112×112 alignment template for the expression model.
        self._reference = np.array(
            [
                (38.2946, 51.6963),
                (73.5318, 51.5014),
                (56.0252, 71.7366),
                (41.5493, 92.3655),
                (70.7299, 92.2041),
            ],
            dtype=np.float32,
        )
        try:
            directory = Path(model_dir)
            detector_bytes = np.frombuffer((directory / FACE_MODEL).read_bytes(), dtype=np.uint8)
            expression_bytes = np.frombuffer((directory / EXPRESSION_MODEL).read_bytes(), dtype=np.uint8)
            self._detector = cv2.FaceDetectorYN.create(
                "onnx", detector_bytes, np.empty(0, dtype=np.uint8), (320, 320), 0.80
            )
            self._expression = cv2.dnn.readNetFromONNX(expression_bytes)
        except (OSError, cv2.error, AttributeError) as error:
            self.error = str(error)
            return
        self.available = True

    def recognize(self, frame_bgr) -> ExpressionObservation | None:
        if not self.available or frame_bgr is None:
            return None
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) < 48:
            return None

        cv2 = self._cv
        np = self._np
        try:
            height, width = frame_bgr.shape[:2]
            if max(width, height) > 480:
                scale = 480 / max(width, height)
                width, height = max(1, round(width * scale)), max(1, round(height * scale))
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(frame_bgr)
            if faces is None or len(faces) != 1:
                return None
            face = faces[0]
            if not np.isfinite(face).all() or min(face[2:4]) < 48:
                return None
            landmarks = face[4:14].reshape(5, 2)
            transform, _ = cv2.estimateAffinePartial2D(landmarks, self._reference, method=cv2.LMEDS)
            if transform is None or not np.isfinite(transform).all():
                return None
            aligned = cv2.warpAffine(frame_bgr, transform, (112, 112))
            blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
            self._expression.setInput(blob, "data")
            scores = self._expression.forward("label").reshape(-1).astype(np.float64)
        except cv2.error as error:
            self.error = str(error)
            return None

        if scores.size != len(EXPRESSION_LABELS) or not np.isfinite(scores).all():
            return None
        self.error = None
        distribution = np.exp(scores - scores.max())
        distribution /= distribution.sum()
        probabilities = dict(zip(EXPRESSION_LABELS, map(float, distribution)))
        label = EXPRESSION_LABELS[int(distribution.argmax())]
        return ExpressionObservation(label, probabilities[label], probabilities, float(face[-1]))


@dataclass(frozen=True)
class DifficultyDecision:
    difficulty: int
    reason: str
    changed: bool
    sample_count: int
    negative_score: float | None


class DifficultyPolicy:
    """Evaluate only the current player turn and lower at most one level per decision."""

    WINDOW_SECONDS = 8.0
    MIN_SAMPLE_INTERVAL = 0.20
    MIN_VALID_SAMPLES = 5
    MIN_EVIDENCE_SECONDS = 1.0
    MIN_EXPRESSION_CONFIDENCE = 0.55
    MAX_SAMPLE_AGE = 1.5
    NEGATIVE_THRESHOLD = 0.70
    MIN_NEGATIVE_AGREEMENT = 0.75
    COOLDOWN_TURNS = 2
    MIN_DIFFICULTY = 1
    MAX_DIFFICULTY = 3

    def __init__(self, base_difficulty: int = 3):
        self._samples: deque[tuple[float, float, float]] = deque()
        self.reset(base_difficulty)

    @property
    def base_difficulty(self) -> int:
        return self._base_difficulty

    @property
    def difficulty(self) -> int:
        return self._difficulty

    def reset(self, base_difficulty: int | None = None) -> None:
        if base_difficulty is not None:
            if type(base_difficulty) is not int or not self.MIN_DIFFICULTY <= base_difficulty <= self.MAX_DIFFICULTY:
                raise ValueError("Difficulty must be an integer from 1 to 3")
            self._base_difficulty = base_difficulty
        self._difficulty = self._base_difficulty
        self._cooldown = 0
        self.clear_evidence()

    def clear_evidence(self) -> None:
        self._samples.clear()
        self._last_timestamp = None

    def observe(self, observation: ExpressionObservation | None, timestamp: float) -> None:
        if not math.isfinite(timestamp) or (
            self._last_timestamp is not None and timestamp <= self._last_timestamp
        ):
            return
        self._last_timestamp = timestamp
        if observation is None:
            self._samples.clear()
            return
        while self._samples and timestamp - self._samples[0][0] > self.WINDOW_SECONDS:
            self._samples.popleft()
        if self._samples and timestamp - self._samples[-1][0] > self.MAX_SAMPLE_AGE:
            self._samples.clear()
        if observation.confidence < self.MIN_EXPRESSION_CONFIDENCE or (
            self._samples and timestamp - self._samples[-1][0] < self.MIN_SAMPLE_INTERVAL
        ):
            return

        negative_mass = sum(observation.probabilities.get(label, 0.0) for label in NEGATIVE_LABELS)
        weight = observation.face_confidence * observation.confidence
        if 0.0 <= negative_mass <= 1.0 and 0.0 < weight <= 1.0:
            self._samples.append((timestamp, negative_mass, weight))

    def decide(self, timestamp: float) -> DifficultyDecision:
        samples = tuple(
            sample for sample in self._samples if 0 <= timestamp - sample[0] <= self.WINDOW_SECONDS
        )
        self.clear_evidence()
        sample_count = len(samples)
        if self._difficulty == self.MIN_DIFFICULTY:
            return DifficultyDecision(self._difficulty, "minimum", False, sample_count, None)
        if self._cooldown:
            self._cooldown -= 1
            return DifficultyDecision(self._difficulty, "cooldown", False, sample_count, None)
        if (
            sample_count < self.MIN_VALID_SAMPLES
            or samples[-1][0] - samples[0][0] < self.MIN_EVIDENCE_SECONDS
            or timestamp - samples[-1][0] > self.MAX_SAMPLE_AGE
        ):
            return DifficultyDecision(self._difficulty, "insufficient_evidence", False, sample_count, None)

        weighted_mass = sum(mass * weight for _, mass, weight in samples)
        score = weighted_mass / sum(weight for _, _, weight in samples)
        agreement = sum(mass >= self.NEGATIVE_THRESHOLD for _, mass, _ in samples) / sample_count
        if score < self.NEGATIVE_THRESHOLD or agreement < self.MIN_NEGATIVE_AGREEMENT:
            return DifficultyDecision(self._difficulty, "unchanged", False, sample_count, score)
        self._difficulty -= 1
        self._cooldown = self.COOLDOWN_TURNS
        return DifficultyDecision(self._difficulty, "lowered", True, sample_count, score)
