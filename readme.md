<!-- markdownlint-disable MD033 -->
# SARIMA Sales Forecasting Platform

Forecasting workflow that takes transactional sales data from daily granularity through a custom SARIMA training loop tracked in MLflow and finally exposes forecasts through a Flask REST API.

> The code lives inside the `code/` directory so you can run everything from a single workspace folder.

## Key Capabilities
- Data exploration notebooks converted to Python scripts for reproducible insight generation.
- Deterministic monthly aggregation (`code/01_data_transformation.py`) that builds segment-level panels for modeling.
- Custom SARIMA implementation with gradient-updated AR/MA terms and MLflow tracking (`code/02_ML_Flow.py`).
- Production-style API (`code/03_EndPoint.py`) with rate limiting, dynamic model loading, and batch inference helpers.
- Ready-to-run smoke tests (`code/04_API_TEST.sh`) plus MLflow artifacts (`code/artifacts/`) for diagnostics.

## Repository Map

| Path | Description |
| --- | --- |
| `code/00_data_exploration.py` | Exploratory analysis of raw daily transactions: outliers, ACF/PACF, and stationarity checks. |
| `code/01_data_transformation.py` | Aggregates daily data into monthly panels per `CUST_STEERING_L3_NAME` and writes `dataset/dataset_monthly_cust_l3_sarima.csv`. |
| `code/02_ML_Flow.py` | End-to-end SARIMA training loop with MLflow logging, comparison against seasonal naive baselines, and future forecasts. |
| `code/03_EndPoint.py` | Flask API that loads MLflow models from `code/mlruns` and serves `/predict` and `/predict/batch`. |
| `code/04_API_TEST.sh` | Sequential `curl` script that hits `/health`, `/models`, `/models/auto-load`, and forecasting endpoints. |
| `dataset/` | Input/output CSVs (`dataset_monthly_cust_l3_sarima.csv`, `sarima_forecasts.csv`). |
| `code/artifacts/` | Saved plots (residuals, accuracy summaries, forecast visualizations). |
| `code/mlruns/` | Local MLflow tracking directory created by the training script. |
| `requirements.txt` | Locked versions for the runtime dependencies. |

## Getting Started

```bash
git clone <repo>
cd proyecto_final
python -m venv .venv
.venv\Scripts\activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

- Python 3.8 is recommended.
- Google Colab was used originally to author the exploration scripts. If you plan to rerun them inside Colab, uncomment the mount logic; locally you can ignore `google.colab`.
- Place `dataset_daily_sales.csv` (raw transactions) inside the `dataset/` folder before running any scripts.

## Data & Modeling Workflow

1. **Exploration** – Run `python code/00_data_exploration.py` to recreate histograms, ACF/PACF heat checks, and stationarity diagnostics. The script expects `dataset/dataset_daily_sales.csv` and writes figures to screen.
2. **Monthly Aggregation** – Execute `python code/01_data_transformation.py` to clean/filter the raw data, split sales vs returns, backfill missing months, and emit `dataset/dataset_monthly_cust_l3_sarima.csv` for each segment.
3. **Model Training & Tracking** – Launch `python code/02_ML_Flow.py` to:
   - Log-transform, double-difference, and fit the custom SARIMA using gradient descent.
   - Compare against a seasonal naive baseline (RMSE/MAPE/R²) and log metrics to MLflow.
   - Produce plots in `code/artifacts/` and summary CSVs (`accuracy_metrics.csv`, `grid_search_results.csv`, etc.).
   - Save future 12-month forecasts per segment to `dataset/sarima_forecasts.csv`.
   - Persist the trained PythonModel into `code/mlruns` for downstream serving.
4. **Forecast Serving** – `python code/03_EndPoint.py` starts a Flask service on `http://0.0.0.0:5001` that can:
   - List all tracked runs (`GET /models`).
   - Load the most recent run for each segment/target (`POST /models/auto-load`).
   - Serve single-target (`POST /predict`) or combined (`POST /predict/batch`) forecasts with configurable horizons.
   - Apply a per-client 5 req/s rate limit.

### Environment Variables for the API

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_HOST` | `0.0.0.0` | Bind address. |
| `API_PORT` | `5001` | Listening port. |
| `API_DEBUG` | `false` | Toggle Flask debug mode. |
| `AUTO_LOAD_MODELS` | `true` | Automatically load the freshest run per segment/target on startup. |
| `MLFLOW_TRACKING_URI` | `./mlruns` | Tracking directory used by both training and serving. |

## Running MLflow UI

Open a second terminal inside `code/` and run:

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

This exposes experiment **SARIMA_Refactored_v3** (training script) and **SARIMA_Refactored_v2** (API default) so you can inspect metrics, parameters, and artifacts visually.

## API Quick Reference

```bash
# Start the server
cd code
python 03_EndPoint.py

# Smoke test every endpoint
sh 04_API_TEST.sh
```

| Endpoint | Method | Body | Notes |
| --- | --- | --- | --- |
| `/health` | GET | – | Basic readiness + loaded model count. |
| `/models` | GET | Optional `segment`, `target` query params | Pulls finished runs from MLflow ordered by recency. |
| `/models/load` | POST | `{"run_id":"...", "segment":"DEALER", "target":"net_sales"}` | Loads a specific run into memory for serving. |
| `/models/auto-load` | POST | `{}` | Automatically loads the newest run per segment/target pair. |
| `/predict` | POST | `{"segment":"DEALER","target":"returns","num_periods":12,"start_date":"2026-01-01"}` | Returns one target stream with calendar-aligned timestamps. |
| `/predict/batch` | POST | `{"segment":"DEALER","num_periods":12}` | Returns both targets and computed `total_net_forecast`. |

See `code/README_API.md` for verbatim request/response samples and additional troubleshooting tips.

## Outputs & Artifacts
- `code/artifacts/residuals_<segment>_<target>.png` – Residual trace and ACF charts for diagnostics.
- `code/artifacts/forecast_<segment>_<target>.png` – Test vs forecast plots.
- `code/artifacts/sarima_forecast_visualization.png` – 12‑month projection overlay for every segment.
- `dataset/sarima_forecasts.csv` – Tabular export of future forecasts (net, returns, and net total).

## Next Steps & Ideas
1. Expand hyper-parameter exploration by wrapping `sarima_fit` inside a sweep utility and logging results to MLflow for model selection.
2. Containerize `03_EndPoint.py` with gunicorn for multi-worker serving and add authentication if sharing outside the lab network.
3. Build automated regression tests (e.g., pytest + golden datasets) over the data transformation script to catch schema drift early.
4. Replace the hand-rolled ARMA optimizer with a torch/jax backend if you need GPU acceleration or second-order optimizers.
