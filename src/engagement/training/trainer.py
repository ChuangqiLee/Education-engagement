"""Config-driven PyTorch trainer with checkpointing and early stopping."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from engagement.models.registry import build_model
from engagement.utils.device import select_device
from engagement.utils.seed import set_seed, worker_seed

from .metrics import classification_metrics


def _output_logits(output):
    return output.logits if hasattr(output, "logits") else output


def _run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=None):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets, predictions = [], []
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = _output_logits(model(images))
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
            batches += 1
    if not batches:
        raise RuntimeError("data loader yielded no batches")
    metrics = classification_metrics(targets, predictions, len(loader.dataset.classes))
    metrics["loss"] = total_loss / batches
    return metrics


def train_from_config(config: Dict[str, Any], resume: Optional[str] = None) -> Dict[str, Any]:
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    training = config["training"]
    seed = int(training.get("seed", 42))
    set_seed(seed)
    device = select_device(training.get("device", "auto"))
    image_size = int(training.get("image_size", 64))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    data_root = Path(config["data_dir"])
    datasets_by_split = {split: datasets.ImageFolder(data_root / split, transform=transform) for split in ("train", "val", "test")}
    expected = list(config["classes"])
    for split, dataset in datasets_by_split.items():
        if set(dataset.classes) != set(expected):
            raise ValueError("%s classes %r differ from config %r" % (split, dataset.classes, expected))
        remap = {dataset.class_to_idx[label]: expected.index(label) for label in expected}
        dataset.target_transform = lambda target, mapping=remap: mapping[target]
        dataset.classes = expected
        dataset.class_to_idx = {label: index for index, label in enumerate(expected)}
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(training.get("batch_size", 32)),
            shuffle=split == "train",
            num_workers=int(training.get("num_workers", 0)),
            worker_init_fn=worker_seed,
            generator=generator,
        )
        for split, dataset in datasets_by_split.items()
    }
    model_config = dict(config["model"])
    name = model_config.pop("name")
    num_classes = int(model_config.pop("num_classes"))
    model = build_model(name, num_classes, **model_config).to(device)
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=float(training.get("learning_rate", 0.001)))
    loss_fn = nn.CrossEntropyLoss()
    start_epoch, best_loss = 0, float("inf")
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload.get("best_loss", best_loss))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    writer_tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer_tb = SummaryWriter(output_dir / "tensorboard")
    except Exception:
        writer_tb = None
    history = []
    patience = int(training.get("patience", 5))
    stale = 0
    max_batches = training.get("max_batches_per_epoch")
    for epoch in range(start_epoch, int(training.get("epochs", 1))):
        train_metrics = _run_epoch(model, loaders["train"], loss_fn, device, optimizer, max_batches)
        val_metrics = _run_epoch(model, loaders["val"], loss_fn, device, None, max_batches)
        row = {"epoch": epoch, **{"train_" + k: v for k, v in train_metrics.items()}, **{"val_" + k: v for k, v in val_metrics.items()}}
        history.append(row)
        scalar_row = {k: v for k, v in row.items() if not isinstance(v, list)}
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scalar_row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(scalar_row)
        if writer_tb:
            for key, value in scalar_row.items():
                if key != "epoch":
                    writer_tb.add_scalar(key, value, epoch)
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": min(best_loss, val_metrics["loss"]),
            "model_name": name,
            "model_config": model_config,
            "classes": expected,
            "provenance": "SYNTHETIC" if (data_root / "SYNTHETIC_DATA_ONLY.txt").exists() else "USER_DATA",
        }
        torch.save(payload, output_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = float(val_metrics["loss"])
            stale = 0
            torch.save(payload, output_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    if writer_tb:
        writer_tb.close()
    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = _run_epoch(model, loaders["test"], loss_fn, device, None, max_batches)
    result = {
        "model": name,
        "device": str(device),
        "epochs_completed": len(history),
        "test_metrics": test_metrics,
        "checkpoint": str(output_dir / "best.pt"),
        "provenance": best["provenance"],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
