import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from engagement.models.registry import build_model


def test_resnet_variants_forward_and_have_different_stems():
    standard = build_model("resnet34_standard", 3)
    compatible = build_model("resnet34_paper_compatible", 3)
    assert standard.conv1.kernel_size == (7, 7)
    assert compatible.conv1.kernel_size == (3, 3)
    assert tuple(standard(torch.zeros(1, 3, 64, 64)).shape) == (1, 3)
    assert tuple(compatible(torch.zeros(1, 3, 64, 64)).shape) == (1, 3)

