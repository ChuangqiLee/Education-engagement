from engagement.data.synthetic import EXPRESSION_CLASSES, make_imagefolder
from engagement.data.validate import validate_imagefolder


def test_synthetic_dataset_is_valid_and_marked(tmp_path):
    root = tmp_path / "expression"
    make_imagefolder(root, EXPRESSION_CLASSES, samples_per_class=3, size=32)
    report = validate_imagefolder(root, EXPRESSION_CLASSES)
    assert report["valid"]
    assert (root / "SYNTHETIC_DATA_ONLY.txt").exists()

