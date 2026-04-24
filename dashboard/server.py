"""
MLAgentBench Reflexion Dashboard — Flask backend.

Start: python dashboard/server.py
Then open: http://localhost:7860
"""
import os
import sys
import json
import uuid
import time
import subprocess
import threading
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS = os.path.join(_PROJECT, "results")
_LOGS    = os.path.join(_PROJECT, "logs")
_PYTHON  = os.path.join(_PROJECT, "implementation", "venv2", "bin", "python")

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

# run_id -> {status, config, started_at, log_buffer, return_code, proc}
_runs: dict = {}
_lock = threading.Lock()

TASKS      = ["house-price", "spaceship-titanic", "vectorization", "feedback"]
CONDITIONS = ["A", "B", "C"]
MODELS     = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "claude-sonnet-4-6",
    "claude-3-5-sonnet-20241022",
    "claude-haiku-4-5-20251001",
    "gpt-4o",
    "gpt-4o-mini",
]

BASELINES = {
    "house-price":       {"value": 30000, "unit": "MAE ($)", "lower_is_better": True},
    "spaceship-titanic": {"value": 0.72,  "unit": "Accuracy", "lower_is_better": False},
    "vectorization":     {"value": None,  "unit": "Speedup",  "lower_is_better": False},
    "feedback":          {"value": 0.50,  "unit": "Macro-F1", "lower_is_better": False},
}


# ─── subprocess reader ───────────────────────────────────────────────────────

def _reader(run_id: str, proc: subprocess.Popen):
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace")
        with _lock:
            _runs[run_id]["log_buffer"].append(line)
    proc.wait()
    with _lock:
        if run_id in _runs:
            _runs[run_id]["status"] = "completed" if proc.returncode == 0 else "failed"
            _runs[run_id]["return_code"] = proc.returncode


# ─── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config")
def config():
    return jsonify({"tasks": TASKS, "conditions": CONDITIONS, "models": MODELS, "baselines": BASELINES})


@app.route("/api/results")
def list_results():
    rows = []
    for cond in CONDITIONS:
        cond_dir = os.path.join(_RESULTS, cond)
        if not os.path.isdir(cond_dir):
            continue
        for fname in sorted(os.listdir(cond_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(cond_dir, fname)) as f:
                    d = json.load(f)
                task = d.get("task", "")
                score = d.get("final_score")
                bl = BASELINES.get(task, {})
                improvement = None
                if bl.get("value") and score is not None:
                    if bl["lower_is_better"]:
                        improvement = round((bl["value"] - score) / bl["value"] * 100, 1)
                    else:
                        improvement = round((score - bl["value"]) / bl["value"] * 100, 1)
                attempts_list = d.get("attempts", [])
                rows.append({
                    "condition":           cond,
                    "task":                task,
                    "model":               d.get("model", ""),
                    "final_score":         score,
                    "baseline":            bl.get("value"),
                    "unit":                bl.get("unit", ""),
                    "improvement_pct":     improvement,
                    "n_attempts":          len(attempts_list),
                    "attempts_to_success": d.get("attempts_to_success"),
                    "hallucination_count": d.get("hallucination_count"),
                    "file":                fname,
                    "path":                f"/api/result/{cond}/{fname}",
                })
            except Exception:
                pass
    return jsonify(rows)


@app.route("/api/result/<condition>/<filename>")
def get_result(condition, filename):
    if condition not in CONDITIONS:
        return jsonify({"error": "invalid condition"}), 400
    path = os.path.join(_RESULTS, condition, filename)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/run", methods=["POST"])
def start_run():
    body      = request.json or {}
    condition = body.get("condition", "A")
    task      = body.get("task", "house-price")
    model     = body.get("model", "gemini-2.5-flash")
    steps     = int(body.get("steps", 30))
    attempts  = int(body.get("attempts", 3))

    if condition not in CONDITIONS:
        return jsonify({"error": f"Invalid condition '{condition}'"}), 400
    if task not in TASKS:
        return jsonify({"error": f"Invalid task '{task}'"}), 400

    run_id = str(uuid.uuid4())[:8]
    cmd = [
        _PYTHON, "-u",
        os.path.join(_PROJECT, "evaluation", "run_experiment.py"),
        "--condition", condition,
        "--task", task,
        "--model", model,
        "--steps", str(steps),
        "--attempts", str(attempts),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_PROJECT}:{os.path.join(_PROJECT, 'implementation', 'MLAgentBench')}"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=_PROJECT,
        env=env,
    )
    t = threading.Thread(target=_reader, args=(run_id, proc), daemon=True)
    t.start()

    with _lock:
        _runs[run_id] = {
            "proc":        proc,
            "thread":      t,
            "status":      "running",
            "config":      {"condition": condition, "task": task, "model": model, "steps": steps, "attempts": attempts},
            "started_at":  datetime.now().isoformat(),
            "log_buffer":  [],
            "return_code": None,
        }

    return jsonify({"run_id": run_id, "status": "running"})


@app.route("/api/stream/<run_id>")
def stream_run(run_id):
    with _lock:
        run = _runs.get(run_id)
    if not run:
        return jsonify({"error": "unknown run_id"}), 404

    def generate():
        cursor = 0
        while True:
            with _lock:
                buf    = run["log_buffer"]
                status = run["status"]
                chunk  = buf[cursor:]
                cursor += len(chunk)

            for line in chunk:
                yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"

            if status in ("completed", "failed", "killed") and not chunk:
                yield f"data: {json.dumps({'type': 'done', 'status': status, 'return_code': run['return_code']})}\n\n"
                break

            if not chunk:
                yield ": heartbeat\n\n"
                time.sleep(0.3)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


@app.route("/api/status")
def all_status():
    with _lock:
        return jsonify([
            {
                "run_id":     rid,
                "status":     r["status"],
                "config":     r["config"],
                "started_at": r["started_at"],
                "log_lines":  len(r["log_buffer"]),
            }
            for rid, r in _runs.items()
        ])


@app.route("/api/log/<run_id>")
def full_log(run_id):
    """Return the entire buffered log for a run (for reconnection / export)."""
    with _lock:
        run = _runs.get(run_id)
    if not run:
        return jsonify({"error": "unknown run_id"}), 404
    return jsonify({"log": "".join(run["log_buffer"]), "status": run["status"]})


@app.route("/api/kill/<run_id>", methods=["POST"])
def kill_run(run_id):
    with _lock:
        run = _runs.get(run_id)
    if not run:
        return jsonify({"error": "unknown run_id"}), 404
    try:
        run["proc"].terminate()
        with _lock:
            _runs[run_id]["status"] = "killed"
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


if __name__ == "__main__":
    os.makedirs(_RESULTS, exist_ok=True)
    os.makedirs(_LOGS, exist_ok=True)
    print("=" * 60)
    print("  MLAgentBench Reflexion Dashboard")
    print("  http://localhost:7860")
    print("=" * 60)
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
