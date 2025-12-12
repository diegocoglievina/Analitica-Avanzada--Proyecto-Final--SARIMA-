HOST="http://127.0.0.1:5001"

curl -sS "$HOST/health"
curl -sS "$HOST/models"
curl -sS -X POST "$HOST/models/auto-load" -H "Content-Type: application/json" -d '{}'
curl -sS "$HOST/models/loaded"
curl -sS -X POST "$HOST/predict" -H "Content-Type: application/json" -d '{"segment":"DEALER","target":"net_sales","num_periods":12,"start_date":"2026-01-01"}'
curl -sS -X POST "$HOST/predict/batch" -H "Content-Type: application/json" -d '{"segment":"DEALER","num_periods":12,"start_date":"2026-01-01"}'
