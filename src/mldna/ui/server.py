"""The FastAPI app. GET routes only; each one is a thin wrapper over UiData."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mlflow.exceptions import MlflowException

from mldna.ui import gitread
from mldna.ui.data import DEFAULT_LIMIT, UiData, UiDataError

STATIC_DIR = Path(__file__).parent / "static"
HOST = "127.0.0.1"


def create_app(workspace_dir="."):
    data = UiData(workspace_dir)
    app = FastAPI(title="mldna", docs_url="/api/docs", redoc_url=None)

    def answer(fn, *args):
        try:
            return fn(*args)
        except (UiDataError, gitread.GitReadError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except MlflowException as e:
            raise HTTPException(status_code=404, detail=e.message)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/summary")
    def summary():
        return data.summary()

    @app.get("/api/runs")
    def runs(limit: int = DEFAULT_LIMIT):
        return answer(data.runs, max(1, min(limit, 1000)))

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        return answer(data.run, run_id)

    @app.get("/api/code-diff")
    def code_diff(old: str, new: str):
        return answer(data.code_diff, old, new)

    @app.get("/api/data-diff")
    def data_diff(old: str, new: str):
        return answer(data.data_diff, old, new)

    @app.get("/api/files")
    def files(commit: str):
        return answer(data.files, commit)

    @app.get("/api/blame")
    def blame(commit: str, path: str):
        return answer(data.blame, commit, path)

    @app.get("/api/models")
    def models():
        return answer(data.models)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def serve(workspace_dir=".", port=8765, open_browser=True):
    import threading
    import webbrowser

    import uvicorn

    app = create_app(workspace_dir)
    url = f"http://{HOST}:{port}/"
    print(f"mldna ui: serving {Path(workspace_dir).resolve()} at {url} (read-only; Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
