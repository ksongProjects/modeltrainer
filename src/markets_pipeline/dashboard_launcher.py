from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web.cli import main as streamlit_main
    except Exception as exc:
        raise SystemExit(
            "Streamlit is not installed. Run `pip install -e .[dev,ui]` to use the dashboard."
        ) from exc

    dashboard_path = Path(__file__).resolve().with_name("dashboard.py")
    sys.argv = ["streamlit", "run", str(dashboard_path)]
    raise SystemExit(streamlit_main())
