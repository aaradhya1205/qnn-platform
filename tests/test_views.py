"""End-to-end smoke tests of the Streamlit pages, run headlessly through the real router."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _errors(at):
    return [e.value[:200] for e in at.exception]


def _button(at, label):
    found = [b for b in at.button if b.label == label or label in b.label]
    assert found, f"no button {label!r}; have {[b.label for b in at.button]}"
    return found[0]


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def test_home_renders(app):
    assert not _errors(app)


def test_research_empty_state_then_full_flow():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    at.switch_page("views/research.py").run()
    assert not _errors(at)
    loads = [b for b in at.button if b.label == "Load this dataset"]
    assert len(loads) == 6
    loads[0].click().run()                         # the messy clinical dataset
    assert not _errors(at)
    _button(at, "Apply recommended").click().run()  # cleaning step
    assert not _errors(at)
    [s for s in at.select_slider if s.key == "pp_runs"][0].set_value(1).run()
    [s for s in at.slider if s.key == "tr_ep"][0].set_value(3).run()
    _button(at, "Run study").click().run()
    assert not _errors(at)
    assert any("Verdict" in m.value for m in at.markdown)
    assert any(d.label.startswith("Download report") for d in at.get("download_button"))


def test_learning_lab_quiz_marks_lesson_complete():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.switch_page("views/learning.py").run()
    assert not _errors(at)
    radio = [r for r in at.radio if r.key == "quiz_l1_q1"][0]
    radio.set_value(radio.options[1]).run()
    [b for b in at.button if b.key == "quiz_l1_q1_btn"][0].click().run()
    assert "l1" in at.session_state["lab_done"]
