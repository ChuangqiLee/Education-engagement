import numpy as np

from engagement.pipeline.engine import EngagementPipeline


def test_demo_pipeline_is_labelled_and_complete():
    pipeline = EngagementPipeline.from_config("configs/pipeline_demo.yaml")
    try:
        records = pipeline.process_frame(np.full((180, 240, 3), 90, dtype=np.uint8), 2.0, "demo")
    finally:
        pipeline.close()
    assert len(records) == 1
    record = records[0]
    assert record.provenance == "DEMO_UNTRAINED"
    assert record.engagement_level in ("G1", "G2", "G3")
    assert len(record.fuzzy_membership) == 3

