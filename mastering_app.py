"""
Compatibility entrypoint for the Mastering view.

The integrated bilingual app now lives in app.py. This file keeps the previous
`streamlit run mastering_app.py` command working and opens Master Validation by
default.
"""

from app import render_app


if __name__ == "__main__":
    render_app(default_mode="master")
