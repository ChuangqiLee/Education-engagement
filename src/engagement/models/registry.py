"""Unified torchvision model registry."""

from typing import Any, Dict


MODEL_NAMES = (
    "resnet34",
    "resnet34_standard",
    "resnet34_paper_compatible",
    "googlenet",
    "alexnet",
    "vgg19",
)


def _weights(enum, pretrained: bool):
    return enum.DEFAULT if pretrained else None


def _freeze_and_open_head(model, head_names):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name in head_names:
        module = getattr(model, name)
        for parameter in module.parameters():
            parameter.requires_grad = True


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    **kwargs: Any
):
    try:
        from torch import nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("torch and torchvision are required for visual models") from exc
    name = name.lower()
    if name in ("resnet34", "resnet34_standard", "resnet34_paper_compatible"):
        model = models.resnet34(weights=_weights(models.ResNet34_Weights, pretrained))
        if name == "resnet34_paper_compatible":
            kernel = int(kwargs.get("stem_kernel_size", 3))
            stride = int(kwargs.get("stem_stride", 1))
            padding = int(kwargs.get("stem_padding", kernel // 2))
            model.conv1 = nn.Conv2d(3, 64, kernel_size=kernel, stride=stride, padding=padding, bias=False)
            if kwargs.get("remove_maxpool", True):
                model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_names = ("fc",)
    elif name == "googlenet":
        if pretrained:
            model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True)
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
        else:
            model = models.googlenet(weights=None, aux_logits=False, init_weights=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_names = ("fc",)
    elif name == "alexnet":
        model = models.alexnet(weights=_weights(models.AlexNet_Weights, pretrained))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        head_names = ("classifier",)
    elif name == "vgg19":
        model = models.vgg19(weights=_weights(models.VGG19_Weights, pretrained))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        head_names = ("classifier",)
    else:
        raise KeyError("unknown model %r; choices: %s" % (name, ", ".join(MODEL_NAMES)))
    if freeze_backbone:
        _freeze_and_open_head(model, head_names)
    model.model_name = name
    model.num_classes = num_classes
    return model
