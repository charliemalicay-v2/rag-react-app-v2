import os
import sys

from uvicorn import run


def setup_project_paths() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ.setdefault("PYTHONPATH", project_root)


if __name__ == "__main__":
    setup_project_paths()
    run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
