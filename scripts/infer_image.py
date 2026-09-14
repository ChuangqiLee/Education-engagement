#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement.io.video_source import read_image
from engagement.pipeline.engine import EngagementPipeline
from engagement.reporting.html_report import write_html_report
from engagement.utils.serialization import write_json, write_records_csv
from engagement.visualization.overlay import draw_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one image")
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/pipeline_demo.yaml")
    parser.add_argument("--output-dir", default="outputs/image")
    args = parser.parse_args()
    import cv2

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = EngagementPipeline.from_config(args.config)
    image = read_image(args.image)
    try:
        records = pipeline.process_frame(image, 0.0, Path(args.image).name)
    finally:
        pipeline.close()
    overlay = draw_records(image, records, demo=pipeline.demo)
    image_path = output / "annotated.png"
    if not cv2.imwrite(str(image_path), overlay):
        raise RuntimeError("failed to write %s" % image_path)
    write_records_csv(records, output / "events.csv")
    write_json(records, output / "events.json")
    write_html_report(records, output / "report.html")
    print(json.dumps({"records": len(records), "annotated": str(image_path), "provenance": "DEMO_UNTRAINED" if pipeline.demo else "MODEL_INFERENCE"}, indent=2))


if __name__ == "__main__":
    main()

