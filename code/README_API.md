# SARIMA Forecasting API

REST API for serving SARIMA models trained with MLflow. This API allows you to load trained models and generate forecasts for different customer segments.

## Quick Start

### 1. Install Dependencies

```bash
pip install flask mlflow pandas numpy
```

### 2. Start the API Server

```bash
python 03_EndPoint.py
```

The server will start on `http://127.0.0.1:5001` by default.

### 3. Auto-Load Models (Optional)

By default, the server auto-loads the best available model for each segment/target combination. To disable this:

```bash
set AUTO_LOAD_MODELS=false
python 03_EndPoint.py
```

## API Endpoints

### Health Check

```
GET /health
```

Returns server status and loaded model count.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-11T22:30:00",
  "loaded_models_count": 6
}
```

---

### List Available Models

```
GET /models
GET /models?segment=DEALER
GET /models?target=net_sales
```

Lists all trained models from MLflow. Supports filtering by segment and target.

**Response:**
```json
{
  "total_count": 6,
  "models": [
    {
      "run_id": "abc123...",
      "run_name": "DEALER_net_sales_p1d1q1_P0D1Q0s12_20251211_220000",
      "segment": "DEALER",
      "target": "net_sales",
      "sarima_order": "(1,1,1)(0,1,0)[12]",
      "metrics": {
        "sarima_RMSE": 3040.65,
        "sarima_MAPE": 34.45,
        "beats_baseline": 0
      }
    }
  ]
}
```

---

### List Loaded Models

```
GET /models/loaded
```

Shows which models are currently loaded for inference.

---

### Load a Specific Model

```
POST /models/load
Content-Type: application/json

{
  "run_id": "abc123def456...",
  "segment": "DEALER",
  "target": "net_sales"
}
```

Loads a specific model by run ID. This replaces any previously loaded model for the same segment/target.

**Required fields:**
- `run_id`: MLflow run ID
- `segment`: One of `DEALER`, `OEM`, `PROFESSIONAL END-USERS`
- `target`: One of `net_sales`, `returns`

---

### Auto-Load Best Models

```
POST /models/auto-load
```

Automatically loads the most recent trained model for each segment/target combination.

---

### Generate Forecast

```
POST /predict
Content-Type: application/json

{
  "segment": "DEALER",
  "target": "net_sales",
  "num_periods": 12,
  "start_date": "2025-11-01"
}
```

Generates a forecast using the loaded model.

**Required fields:**
- `segment`: Customer segment
- `target`: Forecast target (net_sales or returns)

**Optional fields:**
- `num_periods`: Number of months to forecast (default: 12, max: 24)
- `start_date`: Forecast start date (default: next month)

**Response:**
```json
{
  "segment": "DEALER",
  "target": "net_sales",
  "num_periods": 12,
  "forecast_start": "2025-11-01",
  "forecast_end": "2026-10-01",
  "forecasts": [
    {
      "period": 1,
      "date": "2025-11-01",
      "month": "2025-11",
      "forecast_value": 8500.25
    }
  ],
  "model_info": {
    "run_id": "abc123...",
    "loaded_at": "2025-12-11T22:30:00"
  }
}
```

---

### Batch Forecast (All Targets)

```
POST /predict/batch
Content-Type: application/json

{
  "segment": "DEALER",
  "num_periods": 12,
  "start_date": "2025-11-01"
}
```

Generates forecasts for all targets (net_sales and returns) and computes total_net.

**Response:**
```json
{
  "segment": "DEALER",
  "num_periods": 12,
  "forecasts": [
    {
      "period": 1,
      "date": "2025-11-01",
      "month": "2025-11",
      "net_sales_forecast": 8500.25,
      "returns_forecast": 2100.50,
      "total_net_forecast": 6399.75
    }
  ]
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `127.0.0.1` | Server host |
| `API_PORT` | `5001` | Server port |
| `API_DEBUG` | `false` | Enable Flask debug mode |
| `AUTO_LOAD_MODELS` | `true` | Auto-load models on startup |

## Example Usage with cURL

```bash
# Check health
curl http://localhost:5001/health

# List all models
curl http://localhost:5001/models

# List models for a specific segment
curl "http://localhost:5001/models?segment=DEALER"

# Auto-load best models
curl -X POST http://localhost:5001/models/auto-load

# Load a specific model
curl -X POST http://localhost:5001/models/load \
  -H "Content-Type: application/json" \
  -d '{"run_id": "YOUR_RUN_ID", "segment": "DEALER", "target": "net_sales"}'

# Generate forecast
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"segment": "DEALER", "target": "net_sales", "num_periods": 6}'

# Batch forecast
curl -X POST http://localhost:5001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"segment": "DEALER", "num_periods": 12}'
```

## Example Usage with Python

```python
import requests

BASE_URL = "http://localhost:5001"

# Auto-load models
requests.post(f"{BASE_URL}/models/auto-load")

# Generate forecast
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "segment": "DEALER",
        "target": "net_sales",
        "num_periods": 12
    }
)
forecast = response.json()
print(forecast["forecasts"])
```

## Valid Segments and Targets

**Segments:**
- `DEALER`
- `OEM`
- `PROFESSIONAL END-USERS`

**Targets:**
- `net_sales` - Net sales units
- `returns` - Returns units

## Notes

1. Models must be trained using `02_ML_Flow_refactored.ipynb` or `02_ML_Flow_refactored.py` before running the API.

2. The API looks for models in the `./mlruns` directory. Make sure to run the API from the `code` folder.

3. To switch models, simply call `/models/load` with a different `run_id` for the same segment/target combination.

4. The batch endpoint automatically computes `total_net = net_sales - returns` (clipped to non-negative).
