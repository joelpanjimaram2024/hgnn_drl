"""Web dashboard for HGNN-DRL: interactive scheduling algorithm comparison.

Runs experiment configurations against the existing simulation engine and
serves a small HTML + JS UI that plots our results vs. a standard baseline.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.experiments.runner import ExperimentRunner
from src.experiments.metrics_utils import aggregate_metrics, save_results
from src.utils.config import Config, ExperimentConfig

# ---------------------------------------------------------------------------
# App / logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("dashboard")

app = FastAPI(title="HGNN-DRL Dashboard", version="0.1.0")

# Where the runner persists summary files.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    algorithm: str = Field(default="greedy", description="Algorithm to evaluate")
    scenario: str = Field(default="factory", description="Scenario: factory | drone | healthcare")
    n_tasks: int = Field(default=50, description="Number of tasks to schedule")
    seed: int = Field(default=42, description="Deterministic seed")
    horizon: float = Field(default=300.0, description="Simulation horizon in seconds")
    base_algorithm: str = Field(default="cloud-only", description="Baseline to compare against")
    save_results: bool = Field(default=True, description="Write summary CSVs under dashboard/results")

class ConfigRequest(BaseModel):
    workload_size: int = Field(default=50)
    horizon: float = Field(default=3600.0)
    seeds: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    save_results: bool = Field(default=True)
    save_checkpoints: bool = Field(default=False)
    save_plots: bool = Field(default=False)

# ---------------------------------------------------------------------------
# In-flight run state
# ---------------------------------------------------------------------------
class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.done: bool = False

run_state = RunState()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _worker(req: RunRequest) -> None:
    try:
        cfg = Config(
            seed=req.seed,
            workload_size=req.n_tasks,
            horizon=req.horizon,
            # Experiment-specific knobs go in the nested ExperimentConfig.
            experiments=ExperimentConfig(
                seeds=[req.seed],
                workload_size=req.n_tasks,
                save_results=req.save_results,
                save_checkpoints=False,
                save_plots=False,
            ),
        )
        runner = ExperimentRunner(cfg)
        run = runner.run(
            algorithm=req.algorithm,
            scenario=req.scenario,
            n_tasks=req.n_tasks,
            seed=req.seed,
        )

        # Aggregate across the single run (runner.aggregate is also what the
        # comparison endpoint uses) and persist the summaries in the same
        # shape the comparison tables expect.
        summary = runner.aggregate([run])

        # Persist project-native output files (CSV + TXT + JSON).
        save_results(
            summary,
            out_dir=str(RESULTS_DIR),
            scenario=req.scenario,
            algorithm=req.algorithm,
        )

        # Also write a default summary file (same shape the comparison
        # endpoints read) so the web UI can pick up results via /results.
        try:
            save_results(
                summary,
                out_dir=str(RESULTS_DIR),
                scenario="all",
                algorithm="all",
            )
        except Exception:
            pass

        with run_state.lock:
            run_state.result = {
                **run,
                "summary": summary,
                "results_path": str(RESULTS_DIR / "summary_all.csv"),
            }
            run_state.error = None
            run_state.done = True
    except Exception as exc:  # noqa: BLE001
        with run_state.lock:
            run_state.result = None
            run_state.error = f"{type(exc).__name__}: {exc}"
            run_state.done = True



# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    # Seed an empty baseline so the comparison page always has something to show.
    logger.info("Dashboard starting")
    pass


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML, status_code=200)


@app.post("/run")
async def post_run(req: RunRequest) -> JSONResponse:
    # Guard against concurrent runs.
    with run_state.lock:
        if run_state.thread is not None and not run_state.done:
            return JSONResponse({"ok": False, "error": "A run is already in progress"}, status_code=409)

        # Reset state for this run.
        run_state.result = None
        run_state.error = None
        run_state.done = False

    run_state.thread = threading.Thread(target=_worker, args=(req,), daemon=True)
    run_state.thread.start()
    return JSONResponse({"ok": True, "message": "Run started"})


@app.get("/status")
async def status() -> JSONResponse:
    with run_state.lock:
        # The worker writes summary_all.csv (also writes summary_<scenario>.*).
        summary_path = RESULTS_DIR / "summary_all.csv" if RESULTS_DIR.exists() else None
        return JSONResponse(
            {
                "done": run_state.done,
                "error": run_state.error,
                "in_progress": run_state.thread is not None and not run_state.done,
                "summary_path": str(summary_path) if summary_path else None,
            }
        )


@app.get("/results")
async def get_results() -> JSONResponse:
    import json

    import pandas as pd

    path = RESULTS_DIR / "summary_all.csv"
    if not path.exists():
        return JSONResponse(
            {
                "ok": False,
                "error": "No results file written yet. Run an experiment first.",
                "columns": [],
                "rows": [],
            }
        )
    df = pd.read_csv(path)
    return JSONResponse({"ok": True, "columns": list(df.columns), "rows": df.to_dict(orient="records")})


@app.get("/compare")
async def get_compare() -> JSONResponse:
    import json

    import pandas as pd

    # Per-run metrics table so we can group results by algorithm.
    path = RESULTS_DIR / "runs_all.csv"
    if not path.exists():
        return JSONResponse(
            {
                "ok": False,
                "error": "No results file written yet. Run an experiment first.",
                "rows": [],
            }
        )
    df = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for alg in sorted(df["algorithm"].unique()):
        sub = df[df["algorithm"] == alg]
        row = {"algorithm": alg, "runs": int(len(sub))}
        for col in ["n_tasks", "n_completed", "n_deadline_misses", "deadline_violation_rate_pct", "average_latency_s", "median_latency_s", "p95_latency_s", "p99_latency_s", "total_energy_J", "energy_per_task_J", "communication_bytes", "communication_transfers", "throughput_tasks_per_s", "wall_time_s"]:
            if col in df.columns:
                row[col] = round(float(sub[col].mean()), 4)
        rows.append(row)
    return JSONResponse({"ok": True, "rows": rows})


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
DASH_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=DASH_DIR / "static"), name="static")

# ---------------------------------------------------------------------------
# UI HTML
# ---------------------------------------------------------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HGNN-DRL Scheduler Dashboard</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <header class="app-header">
    <h1>HGNN-DRL Scheduler Dashboard</h1>
    <p class="tagline">Compare hypergraph DRL scheduling against baselines on the Mist-Fog-Cloud simulator</p>
  </header>

  <main class="container">
    <section class="card run-card">
      <h2>Run an experiment</h2>
      <form id="run-form" autocomplete="off">
        <div class="grid">
          <div class="field">
            <label for="algorithm">Algorithm</label>
            <select id="algorithm" name="algorithm">
              <option value="hgnn+ppo">HGNN+PPO (Proposed)</option>
              <option value="gcn+ppo">GCN+PPO (Pairwise)</option>
              <option value="greedy" selected>Greedy (baseline)</option>
              <option value="cloud-only">Cloud-only (baseline)</option>
              <option value="mist-only">Mist-only (baseline)</option>
            </select>
          </div>
          <div class="field">
            <label for="scenario">Scenario</label>
            <select id="scenario" name="scenario">
              <option value="factory" selected>Factory</option>
              <option value="drone">Drone</option>
              <option value="healthcare">Healthcare</option>
            </select>
          </div>
          <div class="field">
            <label for="n_tasks">Number of tasks</label>
            <input id="n_tasks" name="n_tasks" type="number" value="50" min="1" max="200" />
          </div>
          <div class="field">
            <label for="seed">Seed</label>
            <input id="seed" name="seed" type="number" value="42" min="0" max="100000" />
          </div>
          <div class="field">
            <label for="horizon">Horizon (seconds)</label>
            <input id="horizon" name="horizon" type="number" value="300" min="1" step="60" />
            <span class="hint-inline">default 300 s = ~1-2 s run time; try 3600 for ~8-10 s.</span>
          </div>
          <div class="field">
            <label for="base_algorithm">Baseline to compare</label>
            <select id="base_algorithm" name="base_algorithm">
              <option value="cloud-only">Cloud-only</option>
              <option value="greedy">Greedy</option>
              <option value="mist-only">Mist-only</option>
            </select>
          </div>
        </div>
        <button type="submit" id="run-btn" class="btn primary">Run</button>
        <p id="run-status" class="hint" role="status"></p>
      </form>
    </section>

    <section class="card results-card" id="results-card" hidden>
      <h2>Results</h2>
      <p class="hint" id="results-hint">Waiting for run to finish&hellip;</p>
      <div class="table-wrap"><table id="results-table"></table></div>
    </section>

    <section class="card compare-card" id="compare-card" hidden>
      <h2>Comparison vs baseline</h2>
      <p class="hint" id="compare-hint">Averaging over seeds.</p>
      <div class="chart-wrap"><canvas id="chart"></canvas></div>
    </section>
  </main>

  <footer class="app-footer">
    <p>HGNN-DRL dashboard &mdash; Mist-Fog-Cloud task scheduling, evaluated on the discrete-event simulator.</p>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="/static/app.js"></script>
</body>
</html>
"""
