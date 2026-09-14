"""Checkpoint-backed image classifier used by inference pipelines."""

from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np

from engagement.utils.device import select_device

from .registry import build_model


class TorchImageClassifier:
    def __init__(
        self,
        model_name: str,
        classes: Sequence[str],
        checkpoint: Union[str, Path],
        device: str = "auto",
        image_size: int = 224,
    ):
        import torch
        from torchvision import transforms

        self.classes = tuple(classes)
        self.device = select_device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_config = payload.get("model_config", {})
        model_config["pretrained"] = False
        self.model = build_model(model_name, len(classes), **model_config)
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def predict(self, image_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        import torch

        rgb = image_bgr[:, :, ::-1].copy()
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probability = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        mapping = {label: float(probability[index]) for index, label in enumerate(self.classes)}
        best = int(np.argmax(probability))
        return self.classes[best], float(probability[best]), mapping


class DemoUntrainedClassifier:
    """Deterministic placeholder that is always marked demo/untrained."""

    def __init__(self, classes: Sequence[str], offset: int = 0):
        self.classes = tuple(classes)
        self.offset = offset

    def predict(self, image_bgr: np.ndarray):
        mean = int(float(np.asarray(image_bgr).mean()))
        selected = (mean + self.offset) % len(self.classes)
        low = 0.35 / max(1, len(self.classes) - 1)
        probabilities = {label: low for label in self.classes}
        probabilities[self.classes[selected]] = 0.65
        return self.classes[selected], 0.65, probabilities
