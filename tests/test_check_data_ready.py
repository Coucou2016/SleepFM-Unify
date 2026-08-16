"""check_data_ready stage / naming tests."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_data_ready",
    _ROOT / "scripts" / "check_data_ready.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_mod)
inspect_dir = _mod.inspect_dir


def test_inspect_exported_synthetic_is_pretrain_ready():
    result = inspect_dir(_ROOT / "data" / "synthetic")
    assert result["pretrain_ready"] is True
    assert result["exported_ready"] is True
    assert result["ready"] is True
    # Exported tree is not "raw PSG download"
    assert result["raw_ready"] is False
    assert result["counts"]["index_json"] == 1


def test_inspect_empty_dir_not_ready(tmp_path):
    result = inspect_dir(tmp_path)
    assert result["raw_ready"] is False
    assert result["pretrain_ready"] is False
    assert result["ready"] is False
