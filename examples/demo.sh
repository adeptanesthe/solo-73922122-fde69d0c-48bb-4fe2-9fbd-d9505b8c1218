#!/usr/bin/env bash
# End-to-end demo against a locally running service (default http://127.0.0.1:8000).
#
# Usage:
#   docker compose up --build -d
#   ./examples/demo.sh
#
# The script creates a 5-sample batch, asks which lab conclusion matters most,
# then demonstrates versioned label replacement and withdrawal.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
BATCH_ID="DEMO-001"

echo "== health =="
curl -sS "$BASE_URL/health"
echo

echo "== create batch (version 1) =="
curl -sS -X POST "$BASE_URL/batches" \
  -H 'Content-Type: application/json' \
  --data-binary "@$(dirname "$0")/batch.json" | tee /tmp/batch-create.json
echo

echo "== review: which lab conclusion matters most =="
curl -sS "$BASE_URL/batches/$BATCH_ID/review" | tee /tmp/batch-review.json
echo

echo "== proactive lab queue, quota 2 (version 1) =="
curl -sS -X POST "$BASE_URL/batches/$BATCH_ID/lab-queue" \
  -H 'Content-Type: application/json' \
  --data '{"version": 1, "quota": 2}' | tee /tmp/batch-queue.json
echo

echo "== same version and quota is an idempotent retry =="
curl -sS -X POST "$BASE_URL/batches/$BATCH_ID/lab-queue" \
  -H 'Content-Type: application/json' \
  --data '{"version": 1, "quota": 2}' | tee /tmp/batch-queue-retry.json
echo
cmp -s /tmp/batch-queue.json /tmp/batch-queue-retry.json \
  && echo "identical queue" \
  || { echo "queues differ!"; exit 1; }

echo "== replace sample 3 conclusion (version 1 -> 2) =="
curl -sS -X PUT "$BASE_URL/batches/$BATCH_ID/samples/3/label" \
  -H 'Content-Type: application/json' \
  --data '{"material": "棉", "version": 1}' | tee /tmp/batch-update.json
echo

echo "== a queue request carrying the old version is rejected with 409 =="
curl -sS -o /tmp/batch-queue-conflict.json -w 'HTTP %{http_code}\n' \
  -X POST "$BASE_URL/batches/$BATCH_ID/lab-queue" \
  -H 'Content-Type: application/json' \
  --data '{"version": 1, "quota": 2}'
cat /tmp/batch-queue-conflict.json
echo

echo "== stale version is rejected with 409 =="
curl -sS -o /tmp/batch-conflict.json -w 'HTTP %{http_code}\n' \
  -X DELETE "$BASE_URL/batches/$BATCH_ID/samples/1/label?version=1"
cat /tmp/batch-conflict.json
echo

echo "== withdraw sample 1 conclusion (version 2 -> 3) =="
curl -sS -X DELETE "$BASE_URL/batches/$BATCH_ID/samples/1/label?version=2" | tee /tmp/batch-delete.json
echo

echo "== review again reflects the new version =="
curl -sS "$BASE_URL/batches/$BATCH_ID/review"
echo

echo "== lab queue reflects the new version and label set =="
curl -sS -X POST "$BASE_URL/batches/$BATCH_ID/lab-queue" \
  -H 'Content-Type: application/json' \
  --data '{"version": 3, "quota": 2}'
echo
