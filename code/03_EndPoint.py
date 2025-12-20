"""
SARIMA Model Inference API

This module provides a REST API for serving SARIMA forecasting models trained with MLflow.

"""

import os
import json
import time
from collections import defaultdict, deque
from datetime import datetime

from flask import Flask, request, jsonify, render_template_string
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np


app = Flask(__name__)


MLFLOW_TRACKING_URI = "./mlruns"
EXPERIMENT_NAME = "SARIMA_Refactored_v3"
VALID_SEGMENTS = ["DEALER", "OEM", "PROFESSIONAL END-USERS"]
VALID_TARGETS = ["net_sales", "returns"]


# ----------------------------
# Simple in-memory rate limiter
# Limit: 5 requests per second per client (by IP)
# ----------------------------
RATE_LIMIT_RPS = 10
RATE_LIMIT_WINDOW_SECONDS = 1.0
_rate_limit_buckets = defaultdict(deque)


def _get_client_ip() -> str:
    # If you later deploy behind a reverse proxy, you may want to trust X-Forwarded-For
    # only if your proxy is correctly configured. For now, use direct remote_addr.
    return request.remote_addr or "unknown"


@app.before_request
def enforce_rate_limit():
    client_ip = _get_client_ip()
    now = time.monotonic()
    bucket = _rate_limit_buckets[client_ip]

    # Drop timestamps outside window
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while bucket and bucket[0] <= cutoff:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_RPS:
        retry_after = max(0.0, RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])) if bucket else RATE_LIMIT_WINDOW_SECONDS
        resp = jsonify(
            {
                "error": "rate_limit_exceeded",
                "message": f"Too many requests. Limit is {RATE_LIMIT_RPS} requests per second."
            }
        )
        resp.status_code = 429
        resp.headers["Retry-After"] = f"{retry_after:.3f}"
        return resp

    bucket.append(now)


class ModelManager:
    def __init__(self):
        self.current_models = {}
        self.available_runs = []
        self.mlflow_client = None
        self._initialize_mlflow()

    def _initialize_mlflow(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.refresh_available_runs()

    def refresh_available_runs(self):
        try:
            experiment = self.mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment is None:
                self.available_runs = []
                return

            runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"]
            )

            self.available_runs = []
            for run in runs:
                run_info = {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                    "segment": run.data.params.get("segment", "unknown"),
                    "target": run.data.params.get("target", "unknown"),
                    "sarima_order": (
                        f"({run.data.params.get('p', '?')},{run.data.params.get('d', '?')},{run.data.params.get('q', '?')})"
                        f"({run.data.params.get('seasonal_P', '?')},{run.data.params.get('seasonal_D', '?')},{run.data.params.get('seasonal_Q', '?')})"
                        f"[{run.data.params.get('s', '?')}]"
                    ),
                    "metrics": {
                        "sarima_RMSE": run.data.metrics.get("sarima_RMSE"),
                        "sarima_MAPE": run.data.metrics.get("sarima_MAPE"),
                        "beats_baseline": run.data.metrics.get("beats_baseline")
                    }
                }
                self.available_runs.append(run_info)
        except Exception as e:
            print(f"Error refreshing runs: {e}")
            self.available_runs = []

    def load_model(self, run_id, segment, target):
        model_key = f"{segment}_{target}"

        try:
            model_uri = f"runs:/{run_id}/sarima_model"
            model = mlflow.pyfunc.load_model(model_uri)

            self.current_models[model_key] = {
                "model": model,
                "run_id": run_id,
                "segment": segment,
                "target": target,
                "loaded_at": datetime.now().isoformat()
            }

            return True, f"Model loaded successfully for {model_key}"
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def get_loaded_models(self):
        result = {}
        for key, value in self.current_models.items():
            result[key] = {
                "run_id": value["run_id"],
                "segment": value["segment"],
                "target": value["target"],
                "loaded_at": value["loaded_at"]
            }
        return result

    def predict(self, segment, target, num_periods):
        model_key = f"{segment}_{target}"

        if model_key not in self.current_models:
            return None, f"No model loaded for {model_key}. Please load a model first."

        try:
            model = self.current_models[model_key]["model"]
            input_df = pd.DataFrame({"forecast_periods": [num_periods]})
            predictions = model.predict(input_df)
            predictions_array = np.asarray(predictions, dtype=float)
            rounded_predictions = np.clip(np.rint(predictions_array), 0, None).astype(int)
            return rounded_predictions.tolist(), None
        except Exception as e:
            return None, f"Prediction failed: {str(e)}"

    def auto_load_best_models(self):
        loaded_count = 0

        for segment in VALID_SEGMENTS:
            for target in VALID_TARGETS:
                matching_runs = [r for r in self.available_runs if r["segment"] == segment and r["target"] == target]

                if matching_runs:
                    best_run = matching_runs[0]
                    success, message = self.load_model(best_run["run_id"], segment, target)
                    if success:
                        loaded_count += 1
                        print(f"Auto-loaded: {segment} {target} (run: {best_run['run_id'][:8]}...)")

        return loaded_count


model_manager = ModelManager()


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "SARIMA Forecasting API",
            "version": "1.0.0",
            "endpoints": {
                "GET /": "This help message",
                "GET /dashboard": "Web dashboard for model management and prediction",
                "GET /models": "List all available trained models",
                "GET /models/loaded": "List currently loaded models",
                "POST /models/load": "Load a specific model for inference",
                "POST /models/auto-load": "Auto-load best models for all segments",
                "POST /predict": "Generate forecast predictions",
                "POST /predict/batch": "Generate forecasts for all targets for a segment",
                "GET /health": "Health check"
            }
        }
    )


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SARIMA Forecasting Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-card: rgba(30, 30, 50, 0.8);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: rgba(255, 255, 255, 0.1);
        }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            background: linear-gradient(135deg, var(--bg-primary) 0%, #16213e 50%, var(--bg-primary) 100%);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border-radius: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        .tab {
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .tab:hover {
            background: var(--bg-card);
            color: var(--text-primary);
        }
        
        .tab.active {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
            box-shadow: 0 0 20px var(--accent-glow);
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .btn {
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: var(--accent);
            color: white;
        }
        
        .btn-primary:hover {
            box-shadow: 0 0 20px var(--accent-glow);
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-label {
            display: block;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        
        .form-control {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-primary);
            font-size: 0.9rem;
            transition: border-color 0.3s ease;
        }
        
        .form-control:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        select.form-control {
            cursor: pointer;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .models-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        
        .models-table th, .models-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        .models-table th {
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .models-table tr:hover {
            background: rgba(255, 255, 255, 0.02);
        }
        
        .metric {
            font-family: 'SF Mono', monospace;
            font-size: 0.8rem;
        }
        
        .tag {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .tag-segment {
            background: rgba(99, 102, 241, 0.2);
            color: #818cf8;
        }
        
        .tag-target {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }
        
        .forecast-results {
            margin-top: 1rem;
        }
        
        .forecast-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        
        .forecast-table th, .forecast-table td {
            padding: 0.6rem 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .forecast-table th {
            background: var(--bg-secondary);
            font-weight: 500;
            text-align: left;
        }
        
        .forecast-value {
            font-weight: 600;
            color: var(--success);
        }
        
        .alert {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        
        .alert-success {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--success);
        }
        
        .alert-error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: var(--error);
        }
        
        .hidden {
            display: none;
        }
        
        .loading {
            opacity: 0.5;
            pointer-events: none;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .loaded-models-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        
        .loaded-model-chip {
            padding: 0.5rem 1rem;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 2rem;
            font-size: 0.8rem;
            color: var(--success);
        }
        
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }
        
        @media (max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
            .tabs {
                flex-wrap: wrap;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">SARIMA Forecasting</div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span id="loaded-count">0 models loaded</span>
            </div>
        </header>
        
        <div class="tabs">
            <button class="tab active" data-tab="models">Models</button>
            <button class="tab" data-tab="load">Load Models</button>
            <button class="tab" data-tab="predict">Predict</button>
        </div>
        
        <div id="alert-container"></div>
        
        <!-- Models Tab -->
        <div class="tab-content active" id="tab-models">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Available Models</h2>
                    <div style="display: flex; gap: 0.5rem;">
                        <select class="form-control" style="width: auto;" id="filter-segment">
                            <option value="">All Segments</option>
                            <option value="DEALER">DEALER</option>
                            <option value="OEM">OEM</option>
                            <option value="PROFESSIONAL END-USERS">PROFESSIONAL END-USERS</option>
                        </select>
                        <select class="form-control" style="width: auto;" id="filter-target">
                            <option value="">All Targets</option>
                            <option value="net_sales">net_sales</option>
                            <option value="returns">returns</option>
                        </select>
                        <button class="btn btn-primary" onclick="refreshModels()">Refresh</button>
                    </div>
                </div>
                <div id="models-container">
                    <div class="empty-state">Click Refresh to load models</div>
                </div>
            </div>
        </div>
        
        <!-- Load Tab -->
        <div class="tab-content" id="tab-load">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Load Models</h2>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Auto-load the best models for each segment/target combination, or manually load a specific model by run ID.
                </p>
                <button class="btn btn-success" onclick="autoLoadModels()">Auto-Load Best Models</button>
                
                <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--border);">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem;">Manual Load</h3>
                    <div class="grid-2">
                        <div class="form-group">
                            <label class="form-label">Run ID</label>
                            <input type="text" class="form-control" id="load-run-id" placeholder="Enter run ID...">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Segment</label>
                            <select class="form-control" id="load-segment">
                                <option value="DEALER">DEALER</option>
                                <option value="OEM">OEM</option>
                                <option value="PROFESSIONAL END-USERS">PROFESSIONAL END-USERS</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Target</label>
                        <select class="form-control" id="load-target">
                            <option value="net_sales">net_sales</option>
                            <option value="returns">returns</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" onclick="loadModel()">Load Model</button>
                </div>
                
                <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--border);">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem;">Currently Loaded</h3>
                    <div id="loaded-models-container">
                        <div class="empty-state" style="padding: 1rem;">No models loaded</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Predict Tab -->
        <div class="tab-content" id="tab-predict">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Generate Forecast</h2>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label class="form-label">Segment</label>
                        <select class="form-control" id="predict-segment">
                            <option value="DEALER">DEALER</option>
                            <option value="OEM">OEM</option>
                            <option value="PROFESSIONAL END-USERS">PROFESSIONAL END-USERS</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Target</label>
                        <select class="form-control" id="predict-target">
                            <option value="net_sales">net_sales</option>
                            <option value="returns">returns</option>
                        </select>
                    </div>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label class="form-label">Number of Periods (1-24)</label>
                        <input type="number" class="form-control" id="predict-periods" value="12" min="1" max="24">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Start Date (optional)</label>
                        <input type="date" class="form-control" id="predict-start-date">
                    </div>
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <button class="btn btn-primary" onclick="generateForecast()">Generate Forecast</button>
                    <button class="btn btn-secondary" onclick="generateBatchForecast()">Batch Forecast (All Targets)</button>
                </div>
                
                <div id="forecast-results" class="forecast-results hidden"></div>
            </div>
        </div>
    </div>
    
    <script>
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
            });
        });
        
        function showAlert(message, type = 'success') {
            const container = document.getElementById('alert-container');
            const alert = document.createElement('div');
            alert.className = 'alert alert-' + type;
            alert.textContent = message;
            container.innerHTML = '';
            container.appendChild(alert);
            setTimeout(() => alert.remove(), 5000);
        }
        
        async function refreshModels() {
            const segment = document.getElementById('filter-segment').value;
            const target = document.getElementById('filter-target').value;
            let url = '/models?';
            if (segment) url += 'segment=' + encodeURIComponent(segment) + '&';
            if (target) url += 'target=' + encodeURIComponent(target);
            
            try {
                const res = await fetch(url);
                const data = await res.json();
                renderModels(data.models);
            } catch (e) {
                showAlert('Failed to fetch models: ' + e.message, 'error');
            }
        }
        
        function renderModels(models) {
            const container = document.getElementById('models-container');
            if (!models || models.length === 0) {
                container.innerHTML = '<div class="empty-state">No models found</div>';
                return;
            }
            
            let html = '<table class="models-table"><thead><tr>';
            html += '<th>Run Name</th><th>Segment</th><th>Target</th><th>Order</th><th>RMSE</th><th>MAPE</th><th>Date</th>';
            html += '</tr></thead><tbody>';
            
            models.forEach(m => {
                html += '<tr>';
                html += '<td><strong>' + (m.run_name || m.run_id.slice(0,8)) + '</strong></td>';
                html += '<td><span class="tag tag-segment">' + m.segment + '</span></td>';
                html += '<td><span class="tag tag-target">' + m.target + '</span></td>';
                html += '<td class="metric">' + m.sarima_order + '</td>';
                html += '<td class="metric">' + (m.metrics.sarima_RMSE ? m.metrics.sarima_RMSE.toFixed(2) : '-') + '</td>';
                html += '<td class="metric">' + (m.metrics.sarima_MAPE ? m.metrics.sarima_MAPE.toFixed(2) + '%' : '-') + '</td>';
                html += '<td style="color: var(--text-secondary); font-size: 0.8rem;">' + m.start_time.slice(0,10) + '</td>';
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        async function autoLoadModels() {
            try {
                const res = await fetch('/models/auto-load', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
                const data = await res.json();
                if (data.success) {
                    showAlert(data.message);
                    updateLoadedModels();
                } else {
                    showAlert(data.error || 'Failed to auto-load', 'error');
                }
            } catch (e) {
                showAlert('Error: ' + e.message, 'error');
            }
        }
        
        async function loadModel() {
            const runId = document.getElementById('load-run-id').value;
            const segment = document.getElementById('load-segment').value;
            const target = document.getElementById('load-target').value;
            
            if (!runId) {
                showAlert('Please enter a run ID', 'error');
                return;
            }
            
            try {
                const res = await fetch('/models/load', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ run_id: runId, segment: segment, target: target })
                });
                const data = await res.json();
                if (data.success) {
                    showAlert(data.message);
                    updateLoadedModels();
                } else {
                    showAlert(data.error || 'Failed to load model', 'error');
                }
            } catch (e) {
                showAlert('Error: ' + e.message, 'error');
            }
        }
        
        async function updateLoadedModels() {
            try {
                const res = await fetch('/models/loaded');
                const data = await res.json();
                document.getElementById('loaded-count').textContent = data.count + ' models loaded';
                
                const container = document.getElementById('loaded-models-container');
                if (data.count === 0) {
                    container.innerHTML = '<div class="empty-state" style="padding: 1rem;">No models loaded</div>';
                } else {
                    let html = '<div class="loaded-models-list">';
                    Object.keys(data.loaded_models).forEach(key => {
                        html += '<span class="loaded-model-chip">' + key + '</span>';
                    });
                    html += '</div>';
                    container.innerHTML = html;
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        async function generateForecast() {
            const segment = document.getElementById('predict-segment').value;
            const target = document.getElementById('predict-target').value;
            const numPeriods = parseInt(document.getElementById('predict-periods').value);
            const startDate = document.getElementById('predict-start-date').value;
            
            const body = { segment, target, num_periods: numPeriods };
            if (startDate) body.start_date = startDate;
            
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                
                if (data.error) {
                    showAlert(data.error, 'error');
                    return;
                }
                
                renderForecast(data);
            } catch (e) {
                showAlert('Error: ' + e.message, 'error');
            }
        }
        
        async function generateBatchForecast() {
            const segment = document.getElementById('predict-segment').value;
            const numPeriods = parseInt(document.getElementById('predict-periods').value);
            const startDate = document.getElementById('predict-start-date').value;
            
            const body = { segment, num_periods: numPeriods };
            if (startDate) body.start_date = startDate;
            
            try {
                const res = await fetch('/predict/batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                
                if (data.error && !data.forecasts) {
                    showAlert(data.error, 'error');
                    return;
                }
                
                renderBatchForecast(data);
            } catch (e) {
                showAlert('Error: ' + e.message, 'error');
            }
        }
        
        function renderForecast(data) {
            const container = document.getElementById('forecast-results');
            container.classList.remove('hidden');
            
            let html = '<h3 style="margin: 1rem 0;">Forecast: ' + data.segment + ' - ' + data.target + '</h3>';
            html += '<p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem;">';
            html += data.forecast_start + ' to ' + data.forecast_end + '</p>';
            html += '<table class="forecast-table"><thead><tr>';
            html += '<th>Period</th><th>Month</th><th>Forecast Value</th>';
            html += '</tr></thead><tbody>';
            
            data.forecasts.forEach(f => {
                html += '<tr>';
                html += '<td>' + f.period + '</td>';
                html += '<td>' + f.month + '</td>';
                html += '<td class="forecast-value">' + f.forecast_value.toLocaleString() + '</td>';
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        function renderBatchForecast(data) {
            const container = document.getElementById('forecast-results');
            container.classList.remove('hidden');
            
            let html = '<h3 style="margin: 1rem 0;">Batch Forecast: ' + data.segment + '</h3>';
            html += '<p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem;">';
            html += data.forecast_start + ' to ' + data.forecast_end + '</p>';
            html += '<table class="forecast-table"><thead><tr>';
            html += '<th>Month</th><th>Net Sales</th><th>Returns</th><th>Net Total</th>';
            html += '</tr></thead><tbody>';
            
            data.forecasts.forEach(f => {
                html += '<tr>';
                html += '<td>' + f.month + '</td>';
                html += '<td class="forecast-value">' + (f.net_sales_forecast ? f.net_sales_forecast.toLocaleString() : '-') + '</td>';
                html += '<td style="color: var(--warning);">' + (f.returns_forecast ? f.returns_forecast.toLocaleString() : '-') + '</td>';
                html += '<td style="color: var(--accent); font-weight: 600;">' + (f.total_net_forecast ? f.total_net_forecast.toLocaleString() : '-') + '</td>';
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        updateLoadedModels();
    </script>
</body>
</html>
"""


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "loaded_models_count": len(model_manager.current_models)
        }
    )


@app.route("/models", methods=["GET"])
def list_models():
    model_manager.refresh_available_runs()

    segment_filter = request.args.get("segment")
    target_filter = request.args.get("target")

    runs = model_manager.available_runs

    if segment_filter:
        runs = [r for r in runs if r["segment"] == segment_filter]
    if target_filter:
        runs = [r for r in runs if r["target"] == target_filter]

    return jsonify(
        {
            "total_count": len(runs),
            "filters_applied": {"segment": segment_filter, "target": target_filter},
            "models": runs
        }
    )


@app.route("/models/loaded", methods=["GET"])
def list_loaded_models():
    return jsonify({"loaded_models": model_manager.get_loaded_models(), "count": len(model_manager.current_models)})


@app.route("/models/load", methods=["POST"])
def load_model():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    run_id = data.get("run_id")
    segment = data.get("segment")
    target = data.get("target")

    if not run_id:
        return jsonify({"error": "run_id is required"}), 400
    if not segment:
        return jsonify({"error": "segment is required"}), 400
    if not target:
        return jsonify({"error": "target is required"}), 400

    if segment not in VALID_SEGMENTS:
        return jsonify({"error": f"Invalid segment. Must be one of: {VALID_SEGMENTS}"}), 400

    if target not in VALID_TARGETS:
        return jsonify({"error": f"Invalid target. Must be one of: {VALID_TARGETS}"}), 400

    success, message = model_manager.load_model(run_id, segment, target)

    if success:
        return jsonify({"success": True, "message": message, "model_key": f"{segment}_{target}"})
    return jsonify({"success": False, "error": message}), 500


@app.route("/models/auto-load", methods=["POST"])
def auto_load_models():
    loaded_count = model_manager.auto_load_best_models()
    return jsonify(
        {
            "success": True,
            "message": f"Auto-loaded {loaded_count} models",
            "loaded_models": model_manager.get_loaded_models()
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    segment = data.get("segment")
    target = data.get("target")
    num_periods = data.get("num_periods", 12)
    start_date = data.get("start_date")

    if not segment:
        return jsonify({"error": "segment is required"}), 400
    if not target:
        return jsonify({"error": "target is required"}), 400

    if segment not in VALID_SEGMENTS:
        return jsonify({"error": f"Invalid segment. Must be one of: {VALID_SEGMENTS}"}), 400

    if target not in VALID_TARGETS:
        return jsonify({"error": f"Invalid target. Must be one of: {VALID_TARGETS}"}), 400

    if not isinstance(num_periods, int) or num_periods < 1 or num_periods > 24:
        return jsonify({"error": "num_periods must be an integer between 1 and 24"}), 400

    predictions, error = model_manager.predict(segment, target, num_periods)
    if error:
        return jsonify({"error": error}), 400

    if start_date:
        try:
            base_date = pd.Timestamp(start_date)
        except Exception:
            base_date = pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=1)
    else:
        base_date = pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=1)

    forecast_dates = pd.date_range(start=base_date, periods=num_periods, freq="MS")

    forecast_result = []
    for i, (date, value) in enumerate(zip(forecast_dates, predictions)):
        units_value = max(int(round(float(value))), 0)
        forecast_result.append(
            {
                "period": i + 1,
                "date": date.strftime("%Y-%m-%d"),
                "month": date.strftime("%Y-%m"),
                "forecast_value": units_value
            }
        )

    return jsonify(
        {
            "segment": segment,
            "target": target,
            "num_periods": num_periods,
            "forecast_start": forecast_dates[0].strftime("%Y-%m-%d"),
            "forecast_end": forecast_dates[-1].strftime("%Y-%m-%d"),
            "forecasts": forecast_result,
            "model_info": {
                "run_id": model_manager.current_models.get(f"{segment}_{target}", {}).get("run_id"),
                "loaded_at": model_manager.current_models.get(f"{segment}_{target}", {}).get("loaded_at")
            }
        }
    )


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    segment = data.get("segment")
    num_periods = data.get("num_periods", 12)
    start_date = data.get("start_date")

    if not segment:
        return jsonify({"error": "segment is required"}), 400

    if segment not in VALID_SEGMENTS:
        return jsonify({"error": f"Invalid segment. Must be one of: {VALID_SEGMENTS}"}), 400

    if not isinstance(num_periods, int) or num_periods < 1 or num_periods > 24:
        return jsonify({"error": "num_periods must be an integer between 1 and 24"}), 400

    results = {}
    errors = {}

    for target in VALID_TARGETS:
        predictions, error = model_manager.predict(segment, target, num_periods)
        if error:
            errors[target] = error
        else:
            results[target] = predictions

    if not results:
        return jsonify({"error": "No predictions available", "details": errors}), 400

    if start_date:
        try:
            base_date = pd.Timestamp(start_date)
        except Exception:
            base_date = pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=1)
    else:
        base_date = pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=1)

    forecast_dates = pd.date_range(start=base_date, periods=num_periods, freq="MS")

    forecast_result = []
    for i, date in enumerate(forecast_dates):
        entry = {"period": i + 1, "date": date.strftime("%Y-%m-%d"), "month": date.strftime("%Y-%m")}
        for target, preds in results.items():
            units_value = max(int(round(float(preds[i]))), 0)
            entry[f"{target}_forecast"] = units_value

        net_sales_units = entry.get("net_sales_forecast")
        returns_units = entry.get("returns_forecast")
        if net_sales_units is not None and returns_units is not None:
            entry["total_net_forecast"] = max(net_sales_units - returns_units, 0)

        forecast_result.append(entry)

    return jsonify(
        {
            "segment": segment,
            "num_periods": num_periods,
            "forecast_start": forecast_dates[0].strftime("%Y-%m-%d"),
            "forecast_end": forecast_dates[-1].strftime("%Y-%m-%d"),
            "forecasts": forecast_result,
            "errors": errors if errors else None
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("SARIMA Forecasting API Server")
    print("=" * 60)

    print("\nInitializing model manager...")
    print(f"Found {len(model_manager.available_runs)} trained models in MLflow")

    auto_load = os.environ.get("AUTO_LOAD_MODELS", "true").lower() == "true"
    if auto_load:
        print("\nAuto-loading best models...")
        loaded = model_manager.auto_load_best_models()
        print(f"Loaded {loaded} models")

    # Bind externally by default so other machines can reach it
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "5001"))
    debug = os.environ.get("API_DEBUG", "false").lower() == "true"

    print(f"\nRate limit: {RATE_LIMIT_RPS} req/s per client")
    print(f"Starting server on http://{host}:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)