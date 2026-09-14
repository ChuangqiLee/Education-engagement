#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement.io.video_source import VideoSource
from engagement.pipeline.engine import EngagementPipeline
from engagement.reporting.html_report import write_html_report
from engagement.utils.serialization import write_json, write_records_csv
from engagement.visualization.overlay import draw_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline sampled video analysis")
    parser.add_argument("video")
    parser.add_argument("--config", default="configs/pipeline_demo.yaml")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", default="outputs/video")
    args = parser.parse_args()
    import cv2

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = EngagementPipeline.from_config(args.config)
    all_records = []
    writer = None
    video_path = output / "annotated.mp4"
    try:
        with VideoSource(args.video) as source:
            for sample_index, packet in enumerate(source.sampled_frames(args.interval)):
                if args.max_samples is not None and sample_index >= args.max_samples:
                    break
                records = pipeline.process_frame(packet.frame, packet.timestamp_seconds, packet.source_id)
                all_records.extend(records)
                overlay = draw_records(packet.frame, records, demo=pipeline.demo)
                if writer is None:
                    height, width = overlay.shape[:2]
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(0.5, 1.0 / args.interval), (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("failed to create annotated video")
                writer.write(overlay)
    finally:
        if writer is not None:
            writer.release()
        pipeline.close()
    write_records_csv(all_records, output / "events.csv")
    write_json(all_records, output / "events.json")
    write_html_report(all_records, output / "report.html")
    print(json.dumps({"records": len(all_records), "annotated_video": str(video_path), "provenance": "DEMO_UNTRAINED" if pipeline.demo else "MODEL_INFERENCE"}, indent=2))


if __name__ == "__main__":
    main()

