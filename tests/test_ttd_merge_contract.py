"""Exercise the production lazy loaders without loading model weights."""
import ast
import os
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from utils.device_env import resolve_sensevoice_device

ROOT = Path(__file__).resolve().parents[1]


def test_ttd_lazy_loader_preserves_cpu_default_and_resolves_auto():
    tree = ast.parse((ROOT / "api.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_sensevoice_model")
    for env, available, expected in [({}, True, "cpu"), ({"SENSEVOICE_DEVICE": "auto"}, False, "cpu"), ({"SENSEVOICE_DEVICE": "auto"}, True, "cuda:0"), ({"SENSEVOICE_DEVICE": "cuda:1"}, False, "cuda:1")]:
        model = Mock()
        factory = Mock(return_value=(model, {"loaded": True}))
        ns = {"os": os, "SenseVoiceSmall": SimpleNamespace(from_pretrained=factory), "resolve_sensevoice_device": lambda requested: resolve_sensevoice_device(requested, cuda_available=available)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "api.py", "exec"), ns)
        factory.assert_not_called()
        with patch.dict(os.environ, env, clear=True):
            assert ns["load_sensevoice_model"]() == (model, {"loaded": True})
        factory.assert_called_once_with(model="iic/SenseVoiceSmall", device=expected)
        model.eval.assert_called_once()


def test_upstream_image_uses_upstream_api_without_private_ttd_imports():
    assert '"upstream_api:app"' in (ROOT / "Dockerfile").read_text()
    assert "ttd_fastapi_utils" not in (ROOT / "upstream_api.py").read_text()
    assert "api:app" in (ROOT / "docker/Dockerfile").read_text()
