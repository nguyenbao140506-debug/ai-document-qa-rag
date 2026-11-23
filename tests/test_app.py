from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_initial_screen_runs_without_model_download():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.metric[0].value == "0"
    assert app.metric[1].value == "0"
    assert any("Upload text PDFs" in notice.value for notice in app.info)
