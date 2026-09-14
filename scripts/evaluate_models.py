#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the unified model registry")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    if not args.forward_only:
        parser.error("only --forward-only is available without a real evaluation dataset")
    import torch
    from engagement.models.registry import MODEL_NAMES, build_model

    results = {}
    for name in MODEL_NAMES:
        model = build_model(name, 3, pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, args.image_size, args.image_size))
        logits = output.logits if hasattr(output, "logits") else output
        results[name] = list(logits.shape)
    print(json.dumps({"provenance": "FORWARD_SMOKE_TEST", "models": results}, indent=2))


if __name__ == "__main__":
    main()

