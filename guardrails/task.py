"""Task definitions: what a guardrail is being asked to detect.

A task is one TOML file in `tasks/` (see `tasks/_template.toml`). It holds the
definition every model-based guardrail must follow (question + criteria), the
human-facing words for the two classes, where the data and results live, and
optional extras for the scorecard (attack families, hard negatives, notes).

    from guardrails.task import load_task, list_tasks
    task = load_task("prompt_injection")
    task.question, task.positive_label, task.data_dir

This module has no dependencies beyond the standard library (`tomllib`).
"""

from __future__ import annotations

import json

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Both are read at call time (tests point them at a tmp dir).
TASKS_DIR = ROOT / "tasks"
DEFAULT_TASK = "prompt_injection"


class TaskError(ValueError):
    """A task file is missing, malformed or incomplete (message is user-facing)."""


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    positive_label: str  # what a guardrail should flag, e.g. "attack"
    negative_label: str  # what it should let through, e.g. "safe prompt"
    # The definition, shared word for word by every model-based guardrail.
    question: str
    criteria_true: str
    criteria_false: str
    judge_role: str  # opening line of the LLM judge's system prompt
    question_key: str  # Jev's question key, e.g. "is_attack"
    state_key: str  # key of the text in Jev's state and in the judge's JSON wrapper
    data_dir: Path
    results_dir: Path
    split_notes: dict[str, str] = field(default_factory=dict)
    families: dict[str, list[str]] = field(default_factory=dict)  # family -> categories
    hard_families: tuple[str, ...] = ()
    hard_negative_categories: tuple[str, ...] = ()
    hard_negative_tags: tuple[str, ...] = ()
    scorecard: dict[str, str] = field(default_factory=dict)  # optional prose, see template

    # ---- words derived from the two labels --------------------------------
    @property
    def pos_plural(self) -> str:
        return self.positive_label + "s"

    @property
    def neg_plural(self) -> str:
        return self.negative_label + "s"

    @property
    def neg_adj(self) -> str:
        """'safe prompt' -> 'safe'; a one-word label is kept as is."""
        words = self.negative_label.split()
        return " ".join(words[:-1]) or self.negative_label

    @property
    def hard_neg_label(self) -> str:
        return f"tricky-but-{self.neg_adj}"

    @property
    def answer_key(self) -> str:
        """Key of the LLM judge's JSON answer: the question key without a leading 'is_'."""
        return self.question_key.removeprefix("is_")

    # ---- data helpers -----------------------------------------------------
    @property
    def has_hard_negatives(self) -> bool:
        return bool(self.hard_negative_categories or self.hard_negative_tags)

    def is_hard_negative(self, label, category, tags) -> bool:
        """A negative that is tricky: its category or one of its |-joined tags is on the task's list."""
        tagset = {t for t in str(tags).split("|") if t}
        return int(label) == 0 and (
            str(category) in self.hard_negative_categories
            or any(t in tagset for t in self.hard_negative_tags)
        )

    def category_to_family(self) -> dict[str, str]:
        return {c: fam for fam, cats in self.families.items() for c in cats}

    def split_note(self, split: str) -> str:
        if split in self.split_notes:
            return self.split_notes[split]
        try:
            where = self.data_dir.relative_to(ROOT)
        except ValueError:
            where = self.data_dir
        return f"{where}/{split}.csv"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def list_tasks(tasks_dir: Path | None = None) -> list[str]:
    """Names of the real tasks (files starting with '_' are templates and skipped)."""
    d = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    return sorted(p.stem for p in d.glob("*.toml") if not p.stem.startswith("_"))


def _text(table: dict, key: str, where: str, path: Path) -> str:
    val = table.get(key)
    if not isinstance(val, str) or not val.strip():
        raise TaskError(f"{path.name}: {where}{key} is required (a non-empty string)")
    return val


def _str_list(val, what: str, path: Path) -> list[str]:
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise TaskError(f"{path.name}: {what} must be a list of strings")
    return list(val)


def _dir(table: dict, key: str, default: str, root: Path, path: Path) -> Path:
    val = table.get(key, default)
    if not isinstance(val, str) or not val.strip():
        raise TaskError(f"{path.name}: {key} must be a path string")
    p = Path(val)
    return p if p.is_absolute() else root / p


def load_task(name: str, tasks_dir: Path | None = None) -> Task:
    """Load `tasks/<name>.toml`. Raises TaskError with a readable message on any problem."""
    d = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    path = d / f"{name}.toml"
    if name.startswith("_") or not path.is_file():
        raise TaskError(f"unknown task {name!r}; choose from {list_tasks(d)}")
    try:
        with open(path, "rb") as f:
            t = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise TaskError(f"{path.name}: not valid TOML ({exc})") from None
    if t.get("name") != name:
        raise TaskError(f"{path.name}: name must be {name!r} (the file name), got {t.get('name')!r}")
    defn = t.get("definition")
    if not isinstance(defn, dict):
        raise TaskError(f"{path.name}: a [definition] table is required")
    fam = t.get("families", {})
    if not isinstance(fam, dict):
        raise TaskError(f"{path.name}: [families] must be a table")
    cats = {k: _str_list(v, f"families.{k}", path) for k, v in fam.items()
            if k not in ("hard_families", "hard_negative_categories", "hard_negative_tags")}
    hard = _str_list(fam.get("hard_families", []), "families.hard_families", path)
    unknown = [h for h in hard if h not in cats]
    if unknown:
        raise TaskError(f"{path.name}: hard_families {unknown} are not defined in [families]")
    splits = t.get("splits", {})
    notes = {s: v["note"] for s, v in splits.items() if isinstance(v, dict) and isinstance(v.get("note"), str)}
    sc = t.get("scorecard", {})
    if not isinstance(sc, dict) or not all(isinstance(v, str) for v in sc.values()):
        raise TaskError(f"{path.name}: [scorecard] must be a table of strings")
    return Task(
        name=name,
        description=_text(t, "description", "", path),
        positive_label=_text(t, "positive_label", "", path),
        negative_label=_text(t, "negative_label", "", path),
        question=_text(defn, "question", "[definition] ", path),
        criteria_true=_text(defn, "criteria_true", "[definition] ", path),
        criteria_false=_text(defn, "criteria_false", "[definition] ", path),
        judge_role=_text(defn, "judge_role", "[definition] ", path),
        question_key=_text(defn, "question_key", "[definition] ", path),
        state_key=_text(defn, "state_key", "[definition] ", path),
        data_dir=_dir(t, "data_dir", f"data/{name}", ROOT, path),
        results_dir=_dir(t, "results_dir", f"results/{name}", ROOT, path),
        split_notes=notes,
        families=cats,
        hard_families=tuple(hard),
        hard_negative_categories=tuple(_str_list(fam.get("hard_negative_categories", []),
                                                 "families.hard_negative_categories", path)),
        hard_negative_tags=tuple(_str_list(fam.get("hard_negative_tags", []),
                                           "families.hard_negative_tags", path)),
        scorecard=dict(sc),
    )


def default_task() -> Task:
    return load_task(DEFAULT_TASK)


def show(path) -> Path:
    """`path` relative to the current directory when it is inside it (for tidy messages)."""
    path = Path(path)
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


def definition_fingerprint(task: Task) -> str:
    """Short hash of everything a definition-following guardrail is sent.

    Cached verdicts are tagged with it so answers produced under an old
    definition are never silently mixed with answers under a new one.
    """
    import hashlib

    parts = [task.question, task.criteria_true, task.criteria_false,
             task.judge_role, task.question_key, task.state_key]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]
