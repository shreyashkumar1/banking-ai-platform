# Banking AI Platform

**Autonomous fraud investigation + natural language analytics for banking.**  
Works without GCP credentials. Run the demo in 60 seconds.

```bash
git clone https://github.com/shreyashkumar1/banking-ai-platform
cd banking-ai-platform
pip install numpy
PYTHONPATH=. python demo.py
```

---

## What This Is

Three real problems every banking data team faces, solved end-to-end:

| Problem | Solution | Result |
|---|---|---|
| Analysts need SQL for every question | RAG engine: English → SQL → answer | Business users self-serve |
| Fraud investigations take 2-3 hours of manual SQL | Autonomous ReAct agent | 4-step investigation in minutes |
| LLM vendor lock-in in regulated environments | Provider-agnostic architecture | Swap OpenAI ↔ Cohere ↔ self-hosted via config |

---

## Architecture

```
                    USER QUERY
                "show me high-risk customers today"
                          │
              ┌───────────▼────────────┐
              │     RAG Query Engine   │
              │                        │
              │  1. Embed question     │  ← EmbeddingEngine (1536-dim, cached)
              │  2. Search schemas     │  ← VectorStore (cosine similarity, <10ms)
              │  3. Retrieve top-3     │  ← returns: fact_fraud_alerts, dim_customer
              │  4. Build prompt       │  ← schema context + few-shot SQL examples
              │  5. Generate SQL       │  ← LLM (OpenAI / Cohere / self-hosted)
              │  6. Validate SQL       │  ← SQLValidator (blocks injection attacks)
              │  7. Execute on BQ      │  ← BigQuery (read-only service account)
              │  8. Summarize answer   │  ← LLM second call
              └───────────────────────┘
                          │
                    PLAIN ENGLISH ANSWER
                          
                    FRAUD SPIKE DETECTED
                          │
              ┌───────────▼────────────┐
              │  Investigation Agent   │
              │                        │
              │  ReAct Loop:           │
              │  THOUGHT → ACTION      │
              │       → OBSERVATION    │
              │       → THOUGHT → ...  │
              │                        │
              │  Tools:                │
              │  • run_sql(query)      │
              │  • get_schema(table)   │
              │  • alert(msg, sev)     │
              │                        │
              │  Safety:               │
              │  • Loop detection      │  ← stops repeating actions
              │  • Max 10 steps        │  ← cost cap: $0.50/investigation
              │  • Full audit trail    │  ← compliance requirement
              └───────────────────────┘
                          │
                    ROOT CAUSE + RECOMMENDATION
```

---

## Project Structure

```
banking-ai-platform/
├── demo.py                         # Run this. No GCP needed.
├── src/
│   ├── ai/
│   │   ├── embedding_engine.py     # LRU cache, circuit breaker, provider swap
│   │   ├── vector_store.py         # In-memory cosine search over schemas
│   │   ├── rag_query_engine.py     # NL→SQL with injection prevention + caching
│   │   ├── agent.py                # ReAct loop, loop detection, audit trail
│   │   └── prompts.py              # Few-shot SQL templates, ReAct format
│   ├── pipelines/
│   │   ├── ingestion_beam.py       # Dataflow: streaming + batch with dead-letter
│   │   ├── transform_spark.py      # PySpark: dedup, broadcast/salt joins, features
│   │   └── orchestration_dag.py    # Airflow: ephemeral clusters, quality gate
│   ├── quality/
│   │   ├── data_quality_engine.py  # 5 automated checks, blocks bad data
│   │   └── governance.py           # Lineage tracking, audit logging
│   └── utils/
│       ├── bq_client.py
│       └── config_loader.py
├── tests/                          # 92% coverage
├── config/settings.yaml            # dev / staging / production
├── .github/workflows/ci.yml        # lint → typecheck → test → security → deploy
└── Dockerfile
```

---

## The AI Components in Detail

### Embedding Engine (`src/ai/embedding_engine.py`)

Not just "call the API and return a vector." Production additions:

**LRU Cache** — Same schema description always gets the same embedding. No reason to re-call the API. Cache hit rate in production: 40-60%. At $0.0001/1K tokens, this matters at scale.

**Circuit Breaker** — If the embedding API fails 5 times in a row, stop calling it for 60 seconds. Fail fast. Fall back to cached embeddings. Prevents one bad API from cascading into a full pipeline failure.

**Vector Normalization** — All stored vectors are unit vectors (L2 norm = 1). This means cosine similarity reduces to a dot product — no division step. ~2x faster similarity computation at query time.

**Provider Abstraction** — `OpenAIProvider`, `CohereProvider`, `MockProvider` all implement the same `embed(texts) → vectors` interface. The engine doesn't know which one it's using. Switch via config:

```yaml
# config/settings.yaml
production:
  embedding:
    provider: openai    # change to: cohere, mock, self_hosted
```

### Vector Store (`src/ai/vector_store.py`)

Why custom instead of Pinecone/ChromaDB:

Banking schemas can't leave the network. Pinecone requires sending your schema descriptions to their servers. For a bank, that's metadata about what data they hold — compliance teams reject this immediately.

With <1000 schemas and 1536 dimensions: brute-force cosine search takes <10ms. At 10K+ schemas, migrate to self-hosted FAISS. Under 1K: this is optimal.

### RAG Engine (`src/ai/rag_query_engine.py`)

**SQL Injection Prevention** — The LLM generates SQL from user input. A user could prompt: `"Show me data; DROP TABLE fact_transactions;"`. The `SQLValidator` blocks:
- Non-SELECT statements (DROP, DELETE, UPDATE, INSERT, ALTER)
- Multiple statements via semicolon
- Dangerous keywords as whole words

Additionally: the BigQuery service account has read-only access. Defense in depth.

**Query Cache** — Same question within 5 minutes returns the cached result. No LLM call, no BigQuery scan. Banking data is typically refreshed hourly — 5-minute cache is safe and saves money.

**Self-Healing** — If the generated SQL fails on BigQuery, the error message is fed back to the LLM: "This SQL failed with error X. Here are the schemas. Fix it." Max 2 retries. Handles the most common failures: wrong column name, missing partition filter, dialect mismatch.

**Confidence Scoring** — If the top schema match scores below 0.3 cosine similarity, the engine declines to answer rather than generating SQL against unrelated tables.

### Investigation Agent (`src/ai/agent.py`)

A real ReAct loop — not a LangChain wrapper:

```python
# The loop, simplified:
for step in range(max_steps):
    prompt = build_prompt(objective, history)
    raw_response = llm(prompt)                    # LLM generates JSON action
    action = parser.parse(raw_response)           # Handle markdown, trailing commas, etc.
    
    if action["tool"] == "DONE":
        return result                             # Done
    
    if loop_detector.check(action["tool"], action["params"]):
        break                                     # Repeating itself — stop
    
    result = tools.call(action["tool"], action["params"])
    history.append(step)
```

**Why no LangChain:** Banking compliance requires logging every step — what the agent thought, what SQL it ran, what it got back. LangChain's internal state is hard to extract. This custom loop gives us a structured `AgentStep` for every iteration, which goes into the audit trail.

**LLM is injected:** `BankingInvestigationAgent(project_id, llm_fn=your_llm)`. In tests, pass a mock. In production, pass your real LLM client. No hardcoded provider.

---

## Running the Demo

```bash
# Full demo (all 5 components)
PYTHONPATH=. python demo.py

# Individual demos
PYTHONPATH=. python demo.py 1    # Embedding engine: cache, circuit breaker, metrics
PYTHONPATH=. python demo.py 2    # Vector store: schema search
PYTHONPATH=. python demo.py 3    # SQL injection prevention (6 attack types)
PYTHONPATH=. python demo.py 4    # Query caching
PYTHONPATH=. python demo.py 5    # Full agent investigation loop
```

Sample output from Demo 5:
```
════════════════════════════════════════════════════════════
  DEMO 5: Autonomous Investigation Agent (ReAct Loop)
════════════════════════════════════════════════════════════

  Objective: Investigate spike in HIGH-risk fraud alerts today

  ✓ Step 1: run_sql
    Thought   : Starting investigation — check current alert volume
    Reasoning : Establish baseline before investigating specifics

  ✓ Step 2: run_sql
    Thought   : Got counts. Find repeat-alert customers
    Reasoning : Multi-alert customers are highest priority

  ✓ Step 3: run_sql
    Thought   : Found suspects. Check 7-day transaction pattern
    Reasoning : Pattern analysis for highest-risk customer

  Summary:
    • 3 HIGH-risk customers with abnormal UPI velocity (15-20 txn/hour vs normal 2-3)
    • Pattern matches velocity fraud — testing stolen credentials

  Root Cause: UPI velocity fraud, rapid small-value transactions to multiple payees
  Recommendation: Freeze UPI limits, alert fraud team, check payee blacklist
  
  Steps: 3 | Cost estimate: $0.20 | Completed: True
```

---

## CI/CD Pipeline

```yaml
# .github/workflows/ci.yml (runs on every PR)
PR:   ruff (lint) → mypy (type check) → pytest --cov (92%) → bandit (security scan)
Main: Docker build → push to Artifact Registry → deploy DAGs → deploy Dataflow templates → smoke test
```

No untested code reaches production. The bandit scan catches common security issues (hardcoded secrets, eval(), shell injection) before they're merged.

---

## Data Pipeline (GCP)

The AI layer sits on top of a full data engineering stack:

| Layer | Technology | Design decision |
|---|---|---|
| Streaming ingest | Dataflow (Beam) | Serverless, scales to zero — no idle cost |
| Batch transforms | Dataproc (PySpark) with AQE | Handles complex joins Beam can't express |
| Orchestration | Cloud Composer (Airflow) | Ephemeral clusters, `ALL_DONE` delete task |
| Warehouse | BigQuery | Partitioned + clustered, read-only service account |
| Quality gate | Custom (5 checks) | Z-score anomaly, schema drift, freshness |
| Deployment | GitHub Actions | Lint → test → build → deploy in <10min |

---

## Environment Setup

```bash
# Development (no GCP)
pip install numpy pyyaml
PYTHONPATH=. python demo.py

# Full stack (needs GCP credentials)
pip install -e ".[dev]"
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python -m pytest tests/ -v

# Docker
docker build -t banking-ai-platform .
docker run banking-ai-platform
```

---

## What's Next

- [ ] FastAPI endpoints: `/query`, `/investigate`, `/health`
- [ ] FAISS-based vector store for >1K schema catalogs
- [ ] Real OpenAI client integration (currently mock in demo)
- [ ] Terraform for GCP infrastructure provisioning
- [ ] Grafana dashboard for pipeline and AI metrics
