"""Guardrail auto-discovery."""

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from guardrails import registry

ROOT = Path(__file__).resolve().parent.parent


def test_discovers_exactly_the_five_methods_in_order():
    infos = registry.discover()
    assert list(infos) == ["regex", "protectai", "jev", "luna", "lakera"]
    expected = {
        "regex": ("Regex", True, []),
        "protectai": ("ProtectAI", True, []),
        "jev": ("Jev", False, ["OPENROUTER_API_KEY"]),
        "luna": ("Luna (GPT-6)", False, ["OPENROUTER_API_KEY"]),
        "lakera": ("Lakera Guard", False, ["LAKERA_API_KEY"]),
    }
    for m, (display, local, keys) in expected.items():
        i = infos[m]
        assert (i.display_name, i.local, i.requires_keys) == (display, local, keys)
        assert i.method == m


def test_discovery_does_not_import_heavy_deps():
    code = (
        "import sys; from guardrails import registry; registry.discover(); "
        "bad=[m for m in ('torch','transformers') if m in sys.modules]; "
        "assert not bad, bad; "
        "assert 'guardrails.small_classifier' not in sys.modules"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_available_reports_missing_keys(monkeypatch):
    monkeypatch.setattr(registry, "load_env", lambda: None)  # ignore the real .env
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("LAKERA_API_KEY", raising=False)
    assert registry.available("jev") == (False, "missing OPENROUTER_API_KEY")
    assert registry.available("lakera") == (False, "missing LAKERA_API_KEY")
    assert registry.available("regex") == (True, "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert registry.available("luna") == (True, "")


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        registry.load("nope")


def test_load_builds_guardrail():
    g = registry.load("regex")
    assert g.name == "regex" and g.check("hello").flagged is False


def test_drop_in_module_is_discovered_and_bad_one_warned(tmp_path, monkeypatch):
    (tmp_path / "toy.py").write_text(
        'METHOD = "toy"\nDISPLAY_NAME = "Toy"\nLOCAL = True\nREQUIRES_KEYS: list[str] = []\n'
        "def make():\n    return None\n")
    (tmp_path / "half.py").write_text('METHOD = "half"\ndef make():\n    return None\n')
    (tmp_path / "_private.py").write_text('METHOD = "p"\n')
    (tmp_path / "helper.py").write_text("X = 1\n")
    monkeypatch.setattr(registry, "GUARDRAILS_DIR", tmp_path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        infos = registry.discover()
    assert list(infos) == ["toy"] and infos["toy"].module == "guardrails.toy"
    assert any("half.py" in str(x.message) for x in w)
