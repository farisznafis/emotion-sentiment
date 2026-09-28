from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    emotion: str
    confidence: float
    scores: dict[str, float]