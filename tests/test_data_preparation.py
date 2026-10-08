import importlib.util
from pathlib import Path
from PIL import Image

spec = importlib.util.spec_from_file_location("prepare_visdrone", "scripts/prepare_visdrone.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_visdrone_class_mapping_clipping_and_ignored_labels(tmp_path):
    annotations = tmp_path / "example.txt"
    annotations.write_text("-10,10,30,20,1,4,0,0\n0,0,10,10,0,1,0,0\n0,0,10,10,1,11,0,0\n200,200,10,10,1,2,0,0\n", encoding="utf-8")
    labels = converter.convert_annotation(annotations, 100, 100).splitlines()
    assert labels == ["3 0.100000 0.200000 0.200000 0.200000"]
