"""The prompt_injection scorecard text must not drift.

tests/fixtures/scorecard_synth_*.md were rendered by analyze.py BEFORE tasks existed,
from the synthetic fixture. Only the run-date line is normalised.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze as an  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("split, threshold_split, golden", [
    ("test", "val", "scorecard_synth_test.md"),
    ("val", "val", "scorecard_synth_val.md"),  # also covers the "cut-off split equals main split" warning
])
def test_scorecard_is_unchanged(split, threshold_split, golden):
    src = FIX / "results_synth.csv"
    res = an.analyze(an.load_results(src), split, threshold_split)
    text = re.sub(r"3\. \*\*Run date\*\*: [0-9-]+", "3. **Run date**: <DATE>", an.render_scorecard(res, src))
    assert text == (FIX / golden).read_text()
