"""Guardrail auto-discovery.

Every `guardrails/*.py` (except `_*.py`, `base.py`, `registry.py`) that defines
module-level `METHOD`, `DISPLAY_NAME`, `LOCAL`, `REQUIRES_KEYS` and a `make()`
function is a guardrail. Metadata is read with `ast`, so discovery never
imports the module: no torch, no model load, no API clients.

`TASKS` (optional list of task names) says which tasks it can be run on:
`["*"]` for task-agnostic methods that follow the task's definition (Jev, Luna),
`["prompt_injection"]` for methods built for one task. Missing means
`["prompt_injection"]`, the only task that existed before tasks did.

    from guardrails import registry
    registry.discover()                    # {"regex": ModuleInfo(...), ...}
    registry.discover(task)                # only the methods that support `task`
    registry.available("jev")              # (False, "missing OPENROUTER_API_KEY")
    registry.load("jev", task)             # a Guardrail instance built for `task`
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

GUARDRAILS_DIR = Path(__file__).resolve().parent
ENV_FILE = GUARDRAILS_DIR.parent / ".env"
_SKIP = {"base", "registry", "task"}
# Stable display order for the built-ins; anything else follows alphabetically.
_PREFERRED_ORDER = ["regex", "protectai", "jev", "luna", "lakera"]
_FIELDS = {"METHOD": str, "DISPLAY_NAME": str, "LOCAL": bool, "REQUIRES_KEYS": list}
_OPTIONAL = {"TASKS": list}
_LEGACY_TASKS = ["prompt_injection"]


@dataclass(frozen=True)
class ModuleInfo:
    method: str
    module: str  # importable name, e.g. "guardrails.jev"
    display_name: str
    local: bool
    requires_keys: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=lambda: list(_LEGACY_TASKS))

    def supports(self, task) -> bool:
        """`task` is a Task or a task name."""
        name = getattr(task, "name", task)
        return "*" in self.tasks or name in self.tasks


def _read_metadata(path: Path) -> tuple[ModuleInfo | None, str | None]:
    """Return (info, None) for a guardrail, (None, reason) for a half-written one,
    (None, None) for a file that is not a guardrail at all."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return None, f"cannot parse ({type(exc).__name__})"
    values: dict[str, object] = {}
    has_make = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "make":
            has_make = True
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and (target.id in _FIELDS or target.id in _OPTIONAL):
            try:
                values[target.id] = ast.literal_eval(value)
            except ValueError:
                return None, f"{target.id} must be a literal value"
    if "METHOD" not in values and not has_make:
        return None, None
    missing = [k for k in _FIELDS if k not in values]
    if missing or not has_make:
        what = missing + ([] if has_make else ["make()"])
        return None, "missing " + ", ".join(what)
    for k, typ in {**_FIELDS, **_OPTIONAL}.items():
        if k in values and not isinstance(values[k], typ):
            return None, f"{k} must be a {typ.__name__}"
    keys = values["REQUIRES_KEYS"]
    if not all(isinstance(x, str) for x in keys):  # type: ignore[union-attr]
        return None, "REQUIRES_KEYS must be a list of strings"
    tasks = values.get("TASKS", _LEGACY_TASKS)
    if not tasks or not all(isinstance(x, str) for x in tasks):  # type: ignore[union-attr]
        return None, "TASKS must be a non-empty list of strings"
    return ModuleInfo(values["METHOD"], f"guardrails.{path.stem}",  # type: ignore[arg-type]
                      values["DISPLAY_NAME"], values["LOCAL"],  # type: ignore[arg-type]
                      list(keys), list(tasks)), None  # type: ignore[arg-type]


def discover(task=None) -> dict[str, ModuleInfo]:
    """Map method name -> ModuleInfo for every guardrail module (cheap, no imports).

    With `task` (a Task or task name), only the methods that support it."""
    found: dict[str, ModuleInfo] = {}
    for path in sorted(GUARDRAILS_DIR.glob("*.py")):
        if path.stem.startswith("_") or path.stem in _SKIP:
            continue
        info, problem = _read_metadata(path)
        if problem:
            warnings.warn(f"guardrails/{path.name} ignored: {problem}", stacklevel=2)
        if info is None:
            continue
        if info.method in found:
            warnings.warn(f"guardrails/{path.name} ignored: METHOD {info.method!r} "
                          f"already used by {found[info.method].module}", stacklevel=2)
            continue
        found[info.method] = info
    ordered = [m for m in _PREFERRED_ORDER if m in found]
    ordered += sorted(m for m in found if m not in _PREFERRED_ORDER)
    return {m: found[m] for m in ordered if task is None or found[m].supports(task)}


def _info(method: str) -> ModuleInfo:
    infos = discover()
    if method not in infos:
        raise KeyError(f"unknown method {method!r}; choose from {list(infos)}")
    return infos[method]


def load_env() -> None:
    """Load the repo's .env (never overrides variables already set)."""
    load_dotenv(ENV_FILE)


def missing_keys(method: str) -> list[str]:
    """Required env vars that are unset or empty (after loading .env)."""
    load_env()
    return [k for k in _info(method).requires_keys if not os.environ.get(k, "").strip()]


def available(method: str) -> tuple[bool, str]:
    """(True, "") if the guardrail can run here, else (False, "missing KEY[, KEY]")."""
    missing = missing_keys(method)
    if missing:
        return False, "missing " + ", ".join(missing)
    return True, ""


def load(method: str, task=None):
    """Import the method's module and build its guardrail via `make(task)`.

    `task` is passed only if the module's `make` takes an argument, so guardrails
    written as `make()` keep working. Raises ValueError if the method does not
    support the task."""
    info = _info(method)
    if task is not None and not info.supports(task):
        raise ValueError(f"{method} does not support task {getattr(task, 'name', task)!r} "
                         f"(it supports {info.tasks})")
    make = importlib.import_module(info.module).make
    if task is not None and inspect.signature(make).parameters:
        return make(task)
    return make()
