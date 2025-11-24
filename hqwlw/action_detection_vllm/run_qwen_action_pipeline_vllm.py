from __future__ import annotations

from models.action_classifier import ActionClassifier
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


if __name__ == "__main__":
    pipeline = create_pipeline()
    pipeline.process_stream(stream_key=f"{settings.redis.stream_prefix}default")
