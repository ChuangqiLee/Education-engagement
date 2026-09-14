#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement.io.multi_camera import MultiCameraSource
from engagement.io.video_source import VideoSource
from engagement.pipeline.engine import EngagementPipeline
from engagement.pipeline.realtime import run_packets
from engagement.reporting.html_report import write_html_report
from engagement.utils.serialization import write_json, write_records_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Single/four-camera real-time interface")
    parser.add_argument("--source", action="append", required=True, help="SOURCE or NAME=SOURCE; repeat for multi-camera")
    parser.add_argument("--config", default="configs/pipeline_demo.yaml")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--output-dir", default="outputs/realtime")
    args = parser.parse_args()
    named = {}
    for index, item in enumerate(args.source):
        if "=" in item:
            name, source = item.split("=", 1)
        else:
            name, source = "camera_%d" % index, item
        named[name] = source
    pipeline = EngagementPipeline.from_config(args.config)
    if len(named) == 1:
        name, source_value = next(iter(named.items()))
        source = VideoSource(source_value, name)
    else:
        source = MultiCameraSource(named)
    try:
        records = run_packets(
            pipeline,
            source.sampled_frames(args.interval),
            display=args.display,
            max_frames=args.max_samples,
        )
    finally:
        source.close()
        pipeline.close()
    output = Path(args.output_dir)
    write_records_csv(records, output / "events.csv")
    write_json(records, output / "events.json")
    write_html_report(records, output / "report.html")
    print(json.dumps({"records": len(records), "sources": list(named), "provenance": "DEMO_UNTRAINED" if pipeline.demo else "MODEL_INFERENCE"}, indent=2))


if __name__ == "__main__":
    main()

