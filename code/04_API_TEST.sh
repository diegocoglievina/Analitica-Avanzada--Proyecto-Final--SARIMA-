# ============================================================
# SARIMA Forecasting API - Test Commands
# ============================================================
# Copy and paste each command into your terminal to test the API.
# Replace the HOST variable if connecting from another device.
# ============================================================

# Set your API host (use your machine's IP for remote access)
HOST="http://127.0.0.1:5001"

# ------------------------------------------------------------
# 1. Health Check - Verify the API is running
# ------------------------------------------------------------
curl -sS "$HOST/health"

# ------------------------------------------------------------
# 2. List Available Models - View all trained models in MLflow
# ------------------------------------------------------------
curl -sS "$HOST/models"

# ------------------------------------------------------------
# 3. Filter Models by Segment
# ------------------------------------------------------------
curl -sS "$HOST/models?segment=OEM"

# ------------------------------------------------------------
# 4. Filter Models by Target
# ------------------------------------------------------------
curl -sS "$HOST/models?target=returns"

# ------------------------------------------------------------
# 5. Auto-Load Best Models - Load the best model for each segment/target
# ------------------------------------------------------------
curl -sS -X POST "$HOST/models/auto-load" -H "Content-Type: application/json" -d '{}'

# ------------------------------------------------------------
# 6. Check Loaded Models - See which models are currently active
# ------------------------------------------------------------
curl -sS "$HOST/models/loaded"

# ------------------------------------------------------------
# 7. Load a Specific Model (replace RUN_ID with actual run ID)
# ------------------------------------------------------------
curl -sS -X POST "$HOST/models/load" -H "Content-Type: application/json" -d '{"run_id":"YOUR_RUN_ID_HERE","segment":"OEM","target":"returns"}'

# ------------------------------------------------------------
# 8. Generate Prediction - Single target forecast
# ------------------------------------------------------------
curl -sS -X POST "$HOST/predict" -H "Content-Type: application/json" -d '{"segment":"OEM","target":"returns","num_periods":12,"start_date":"2026-01-01"}'

# ------------------------------------------------------------
# 9. Batch Prediction - All targets for a segment
# ------------------------------------------------------------
curl -sS -X POST "$HOST/predict/batch" -H "Content-Type: application/json" -d '{"segment":"OEM","num_periods":12,"start_date":"2026-01-01"}'

# ------------------------------------------------------------
# 10. Test Other Segments
# ------------------------------------------------------------
curl -sS -X POST "$HOST/predict" -H "Content-Type: application/json" -d '{"segment":"DEALER","target":"net_sales","num_periods":6}'

curl -sS -X POST "$HOST/predict" -H "Content-Type: application/json" -d '{"segment":"PROFESSIONAL END-USERS","target":"net_sales","num_periods":6}'