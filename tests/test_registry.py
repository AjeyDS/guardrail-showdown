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


# ---- tasks -------------------------------------------------------------------
def test_tasks_metadata_of_the_builtins():
    infos = registry.discover()
    assert {m: i.tasks for m, i in infos.items()} == {
        "regex": ["prompt_injection"], "protectai": ["prompt_injection"], "lakera": ["prompt_injection"],
        "jev": ["*"], "luna": ["*"],
    }


def test_discover_filters_by_task():
    assert list(registry.discover("prompt_injection")) == ["regex", "protectai", "jev", "luna", "lakera"]
    assert list(registry.discover("toxicity")) == ["jev", "luna"]  # task-agnostic ones only
    from guardrails.task import load_task
    assert list(registry.discover(load_task("prompt_injection"))) == list(registry.discover())


def test_missing_tasks_means_prompt_injection_and_bad_tasks_warn(tmp_path, monkeypatch):
    base = 'METHOD = "{m}"\nDISPLAY_NAME = "T"\nLOCAL = True\nREQUIRES_KEYS: list[str] = []\n{extra}def make():\n    return None\n'
    (tmp_path / "old.py").write_text(base.format(m="old", extra=""))
    (tmp_path / "any.py").write_text(base.format(m="any", extra='TASKS: list[str] = ["*"]\n'))
    (tmp_path / "bad.py").write_text(base.format(m="bad", extra='TASKS = "toxicity"\n'))
    monkeypatch.setattr(registry, "GUARDRAILS_DIR", tmp_path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert list(registry.discover("prompt_injection")) == ["any", "old"]
        assert list(registry.discover("toxicity")) == ["any"]
    assert any("bad.py" in str(x.message) and "TASKS" in str(x.message) for x in w)


def test_load_passes_the_task_only_to_makers_that_take_it(tmp_path, monkeypatch):
    (tmp_path / "plain.py").write_text(
        'METHOD = "plain"\nDISPLAY_NAME = "P"\nLOCAL = True\nREQUIRES_KEYS: list[str] = []\nTASKS: list[str] = ["*"]\n'
        "def make():\n    return 'no-arg'\n")
    (tmp_path / "taskful.py").write_text(
        'METHOD = "taskful"\nDISPLAY_NAME = "T"\nLOCAL = True\nREQUIRES_KEYS: list[str] = []\nTASKS: list[str] = ["*"]\n'
        "def make(task=None):\n    return ('got', getattr(task, 'name', None))\n")
    monkeypatch.setattr(registry, "GUARDRAILS_DIR", tmp_path)
    monkeypatch.setattr(registry.importlib, "import_module", lambda name: _load(tmp_path, name))
    from guardrails.task import load_task
    task = load_task("prompt_injection")
    assert registry.load("plain", task) == "no-arg"
    assert registry.load("taskful", task) == ("got", "prompt_injection")
    assert registry.load("taskful") == ("got", None)


def _load(tmp_path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, tmp_path / (name.split(".")[-1] + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_rejects_a_task_the_method_does_not_support():
    with pytest.raises(ValueError, match="does not support task 'toxicity'"):
        registry.load("regex", "toxicity")
