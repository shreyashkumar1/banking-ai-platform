# Operations Runbook — Banking AI Platform

## Monitoring & Alerting

### Pipeline Health Checks

| Check | Frequency | Alert Channel | Severity |
|-------|-----------|---------------|----------|
| Airflow DAG status | Every run | Slack #data-alerts | Critical if failed |
| BigQuery freshness | Hourly | Slack + PagerDuty | Critical if > 6 hours |
| Data quality gates | Every load | Slack #data-alerts | Critical/Warning |
| Vector store health | Daily | Slack #ai-ops | Warning |
| LLM API latency | Per request | Datadog | Warning if > 5s |
| Cost monitoring | Daily | Slack #cost-alerts | Warning if > budget |

### Common Issues & Fixes

#### 1. Spark OOM (OutOfMemoryError)

**Symptoms:** Dataproc job fails with `java.lang.OutOfMemoryError: Java heap space`

**Root Cause:** Usually data skew or broadcast table too large

**Fix:**
```bash
# Check data distribution
SELECT customer_id, COUNT(*) as cnt
FROM fact_transactions
GROUP BY 1 ORDER BY 2 DESC LIMIT 10;

# If skewed: enable AQE skew handling
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# If broadcast too large: disable broadcast
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

# If general OOM: increase executor memory
spark.conf.set("spark.executor.memory", "12g")
```

#### 2. BigQuery Query Cost Spike

**Symptoms:** Daily cost alert triggered

**Fix:**
```sql
-- Find expensive queries
SELECT user_email, query, total_bytes_billed / POW(1024,3) as gb_billed
FROM `region-asia-south1`.INFORMATION_SCHEMA.JOBS
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
ORDER BY total_bytes_billed DESC LIMIT 10;

-- Usually: missing partition filter or SELECT *
-- Fix: enforce require_partition_filter = true
```

#### 3. RAG Generating Wrong SQL

**Symptoms:** User reports incorrect query results

**Fix:**
1. Check vector store search results — are the right schemas being retrieved?
2. If wrong schemas: re-index with richer descriptions
3. If right schemas but wrong SQL: add the failing case as a few-shot example
4. If persistent: check if schema has changed — re-embed

#### 4. Agent Infinite Loop

**Symptoms:** Agent investigation takes > 5 minutes or hits max steps

**Fix:**
1. Check step history — is the agent repeating the same SQL?
2. If yes: the objective is too vague. Rephrase with specific scope.
3. If no: the agent is exploring a genuinely complex issue. Review steps manually.

---

## Deployment Procedures

### Deploying New DAGs
```bash
# CI/CD handles this automatically on merge to main
# Manual deploy (emergency only):
gsutil -m rsync -r src/pipelines/ gs://COMPOSER_BUCKET/dags/
```

### Updating Schema Index
```bash
# After adding/modifying BigQuery tables:
python -m src.ai.vector_store --reindex
```

### Rotating LLM API Keys
```bash
# Update GitHub secret
gh secret set OPENAI_API_KEY --body "new-key-here"

# Update GCP Secret Manager
gcloud secrets versions add llm-api-key --data-file=key.txt
```

---

## Incident Response

### Severity Levels

| Level | Definition | Response Time | Example |
|-------|-----------|---------------|---------|
| P0 | Data loss or security breach | 15 min | BigQuery table deleted |
| P1 | Pipeline blocked, no data flowing | 30 min | DAG failing, DQ gate blocking |
| P2 | Degraded performance | 2 hours | Spark job slow, high latency |
| P3 | Minor issue | Next business day | Dashboard formatting, non-critical alert |
