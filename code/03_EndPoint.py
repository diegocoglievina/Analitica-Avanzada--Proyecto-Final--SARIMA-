"""
SARIMA Model Inference API

This module provides a REST API for serving SARIMA forecasting models trained with MLflow.
It allows users to:
- List available trained models
- Select/switch the active model for inference
- Generate forecasts for specific segments and date ranges
"""

import os
import json
import time
from collections import defaultdict, deque
from datetime import datetime

from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np


app = Flask(__name__)


MLFLOW_TRACKING_URI = "./mlruns"
EXPERIMENT_NAME = "SARIMA_Refactored_v2"
VALID_SEGMENTS = ["DEALER", "OEM", "PROFESSIONAL END-USERS"]
VALID_TARGETS = ["net_sales", "returns"]


# ----------------------------
# Simple in-memory rate limiter
# Limit: 5 requests per second per client (by IP)
# ----------------------------
RATE_LIMIT_RPS = 5
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
            return predictions.tolist(), None
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
        forecast_result.append(
            {
                "period": i + 1,
                "date": date.strftime("%Y-%m-%d"),
                "month": date.strftime("%Y-%m"),
                "forecast_value": round(float(value), 2)
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
            entry[f"{target}_forecast"] = round(float(preds[i]), 2)

        if "net_sales" in results and "returns" in results:
            entry["total_net_forecast"] = round(max(float(results["net_sales"][i]) - float(results["returns"][i]), 0.0), 2)

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