from __future__ import annotations

"""Legacy entrypoint kept for backward compatibility.

Core logic has been modularized into:
- models.detection.PersonDetector
- models.action_classifier.ActionClassifier
- pipeline.core.PosePipeline
"""

from models.action_classifier import QwenAction, ActionClassifier
from models.detection import PersonDetector
from pipeline.core import PosePipeline
from config.settings import settings


def create_pipeline() -> PosePipeline:
    detector = PersonDetector(
        model_path=settings.yolo.model_path,
        device=settings.yolo.device,
        conf=settings.yolo.conf,
    )
    classifier = ActionClassifier(batch_size=settings.vllm.max_workers)
    return PosePipeline(detector=detector, classifier=classifier)


__all__ = ["QwenAction", "create_pipeline", "PosePipeline"]
