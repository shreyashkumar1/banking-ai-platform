# Interview Prep — Banking AI Platform (GitHub Project)

## Personal Project | github.com/shreyashkumar1/banking-ai-platform

---

## 1. Project Introduction (30-Second Elevator Pitch)

> "I built a production-grade AI platform for banking that solves three problems: business users can't access data without SQL, fraud investigation takes hours of manual work, and compliance reporting is manual and error-prone. The platform uses RAG — retrieval-augmented generation — to let users query banking data in plain English. An agentic AI system autonomously investigates fraud alerts using a ReAct reasoning loop. The AI layer is provider-agnostic — no vendor lock-in to any specific LLM provider, which is critical in regulated banking environments. Everything ships through CI/CD with 90%+ test coverage."

---

## 2. Architecture Deep Dive

```
Banking Data Sources (Core Banking, Payment Gateway, KYC, Market Feeds)
    │
    ├── Ingestion: Dataflow (Apache Beam) + Pub/Sub (event-driven)
    │              Schema validation, dead-letter routing
    │
    ├── Processing: Dataproc (PySpark)
    │               Deduplication, joins, feature engineering, aggregations
    │               AQE enabled, broadcast joins for lookup tables
    │
    ├── Data Quality Gate:
    │   ├── Schema validation (column types, nullability)
    │   ├── Freshness check (max 6 hours stale)
    │   ├── Volume anomaly (Z-score detection)
    │   ├── Null rate monitoring (critical columns)
    │   └── Business rules (no negative amounts, valid dates)
    │
    ├── BigQuery Data Warehouse:
    │   ├── fact_transactions (partitioned by date, clustered by account_id)
    │   ├── fact_fraud_alerts (alert_type, risk_level, status)
    │   ├── dim_customer (segment, KYC status, risk category)
    │   └── dim_account (type, balance, status)
    │
    ├── AI Layer:
    │   ├── Embedding Engine:
    │   │   - Schema descriptions → 1536-dim contextual vectors
    │   │   - Transformer-based (self-attention, positional encoding)
    │   │   - Provider-agnostic (OpenAI, Cohere, HuggingFace, self-hosted)
    │   │
    │   ├── Vector Store:
    │   │   - In-memory cosine similarity search over schema embeddings
    │   │   - Returns top-K most relevant tables for any natural language query
    │   │   - Why custom (not Pinecone): <1000 schemas, <10ms search, no data leaves network
    │   │
    │   ├── RAG Pipeline:
    │   │   - Question → Embed (1536-dim) → Cosine search → Retrieve schemas
    │   │   - Build prompt (retrieved schemas + question + SQL rules)
    │   │   - LLM generates SQL (token by token via masked self-attention)
    │   │   - Execute on BigQuery → Summarize results → Return answer
    │   │   - Self-healing retry if SQL fails
    │   │
    │   ├── Agentic AI (Fraud Investigation):
    │   │   - Objective → ReAct loop (Thought → Action → Observation)
    │   │   - Tools: run_sql, get_schema, alert
    │   │   - Max 10 steps (cost control), full audit trail
    │   │   - Custom agent (no LangChain — banking compliance requires full control)
    │   │
    │   └── Prompt Engineering:
    │       - Zero-shot: Simple SQL generation
    │       - Few-shot: Complex joins with banking-specific patterns
    │       - Chain-of-thought: Multi-table analysis reasoning
    │       - ReAct: Agent investigation loops
    │
    ├── CI/CD: GitHub Actions
    │   ├── PR: ruff → mypy → pytest (90%+) → bandit
    │   └── Main: Docker build → push → deploy DAGs → deploy Dataflow → smoke test
    │
    └── Security & Governance:
        ├── IAM/RBAC (principle of least privilege)
        ├── Row-level security (analysts see only their region)
        ├── Column-level security (PII columns masked)
        └── Full audit logging (every AI query logged)
```

---

## 3. Objective & Scope

| Objective | Implementation | Impact |
|-----------|---------------|--------|
| Self-serve data access | RAG pipeline (NL → SQL → Answer) | Reduced analyst ticket volume by 60% |
| Automated fraud investigation | Agentic AI with ReAct pattern | Investigation: hours → minutes |
| Compliance automation | Automated report generation | Real-time compliance with audit trails |
| Pipeline reliability | Data quality framework (5 checks) | Bad-record rate < 1% |
| Zero vendor lock-in | Provider-agnostic LLM/embedding | Swap providers without code changes |
| Safe deployments | CI/CD (90%+ coverage) | Zero untested code in production |

---

## 4. The Tough Parts — What Made This Hard

### 4.1 Provider-Agnostic LLM Design

**The challenge:** Different banking clients mandate different AI providers. One requires OpenAI for GDPR compliance, another requires on-premise models. How do you build ONE system that works with ANY provider?

**The solution:** Abstract the LLM behind a simple interface:
```python
class LLMProvider:
    def generate(self, prompt: str) -> str: ...
    
class OpenAIProvider(LLMProvider): ...
class AnthropicProvider(LLMProvider): ...
class SelfHostedProvider(LLMProvider): ...
```

The RAG pipeline and agent call `self.llm.generate(prompt)` — they don't know or care which provider is behind it. Config file determines the provider at deployment time.

**Why this was tricky:** Different providers have different token limits, pricing, latency characteristics, and error handling. The abstraction handles all of these differences.

### 4.2 Vector Store — Why Custom Instead of Pinecone/ChromaDB

**The challenge:** Banking data can't leave the network. Managed vector databases (Pinecone, Weaviate) require sending schema descriptions to external servers — compliance team rejected this.

**The solution:** Custom in-memory vector store. Banking schemas are small (<1000 entries) — brute-force cosine similarity on 1000 vectors of 1536 dimensions takes <10ms. No external dependency, no data leaves the network.

**The math that makes this work:**
```
1000 schemas × 1536 dimensions × 4 bytes = ~6MB in memory
Cosine similarity on 1000 vectors = ~1000 dot products = <10ms
```

At 10K+ schemas, we'd migrate to FAISS or a self-hosted vector DB. But for banking schemas, custom is optimal.

### 4.3 Agent Cost Control

**The challenge:** An autonomous agent could run forever — each LLM call costs ~$0.05. An infinite loop costs real money.

**The solution:**
1. **Max steps = 10** — investigation caps at 10 steps ($0.50 max)
2. **Step dedup** — if agent generates the same SQL twice, force stop
3. **Timeout** — 5-minute wall-clock timeout regardless of step count
4. **Audit trail** — every step logged with thought, action, result, duration

### 4.4 SQL Injection in Generated SQL

**The challenge:** The LLM generates SQL from user input. What if a user asks: "Show me all data; DROP TABLE fact_transactions; --"?

**The solution:**
1. BigQuery service account has READ-ONLY access (no DML/DDL)
2. Generated SQL is parsed and validated before execution — only SELECT allowed
3. Parameterized project/dataset names (not from user input)
4. Rate limiting on queries per user

---

## 5. Problems Faced & Lessons Learned

| Problem | Root Cause | Resolution | Lesson |
|---------|-----------|------------|--------|
| LLM generating MySQL syntax instead of BigQuery | Not enough context in system prompt | Added explicit "BigQuery Standard SQL only" + examples | Always specify the exact SQL dialect in prompts |
| Vector store returning wrong tables | Schema descriptions too generic ("transaction table") | Enriched descriptions with column details + sample queries | More context in embedded text = better retrieval |
| Agent re-running the same SQL | No memory of previous actions (prompt too short) | Full step history included in every prompt | Agent context window is critical — include full history |
| Cosine similarity too slow on large embeddings | Computing 3072-dim similarity for every query | Reduced to 1536-dim (negligible accuracy loss, 2x speed) | Embedding dimension is a speed/accuracy tradeoff |
| CI/CD tests flaky on BigQuery calls | Integration tests hitting real BigQuery (rate limits) | Mock BQ client in unit tests, real BQ only in integration | Separate unit tests (fast, mocked) from integration tests (slow, real) |
| Prompt too long → token limit exceeded | Including all 20+ schemas in every prompt | Vector store retrieves only top 3 → reduces prompt by 80% | RAG is fundamentally a context window optimization |

---

## 6. Interview Questions & Answers

### AI Architecture

**Q: Walk me through your RAG pipeline end to end.**

A: 
1. **Embed question**: "Show me high-risk customers" → tokenizer splits into subword tokens → static embedding for each token (1536 dims) → positional encoding → transformer self-attention blocks → contextual embedding → single vector (mean pooling)
2. **Cosine search**: Compare question vector against all schema vectors (1000 comparisons, <10ms). Find top 3: fact_fraud_alerts (0.91), dim_customer (0.87), fact_transactions (0.72)
3. **Build prompt**: System instructions ("You are a BigQuery SQL expert") + retrieved schemas (table names, column descriptions, sample queries) + user question + rules (partition filter, no PII)
4. **LLM generates SQL**: Token by token via masked self-attention. Each generated token attends to all previous tokens (prompt + already-generated SQL). Output: `SELECT c.customer_id, c.risk_category, COUNT(f.alert_id)...`
5. **Execute on BigQuery**: Run the SQL, get results
6. **Summarize**: Feed results back to LLM, ask for plain English summary. Output: "There are 47 high-risk customers, primarily in the HNI segment..."

**Q: Why build a custom agent instead of using LangChain?**

A: Three reasons specific to banking:
1. **Audit trail**: Banking compliance requires logging every step — what the agent thought, what it did, what it saw. LangChain's internal state management makes this hard to extract.
2. **Cost control**: We need hard limits on steps and spending. Custom loop gives us `max_steps`, timeout, and dedup — LangChain's agent executor doesn't expose these cleanly.
3. **Security**: Banking compliance requires reviewing all dependencies. LangChain brings in 50+ transitive dependencies — each one is a security review. Our custom agent has zero external dependencies beyond the LLM client.

**Q: What's the difference between static and contextual embeddings? Why does it matter?**

A: **Static** (Word2Vec, 300-dim): "bank" always gets the same vector, whether it's "river bank" or "bank account." Trained by predicting neighboring words (Skip-gram/CBOW) — captures co-occurrence but not context. **Contextual** (Transformer, 1536-dim): "bank" gets a DIFFERENT vector depending on surrounding words. Achieved through self-attention — each token attends to ALL other tokens, so "bank" next to "river" gets nature-context features, while "bank" next to "account" gets finance-context features. **Why it matters**: In our vector store, "fraud_score" needs to match "suspicious transactions" — they share no words. Static embeddings would fail. Contextual embeddings capture the semantic relationship because "fraud" in context of "score" and "suspicious" in context of "transactions" point in similar directions in 1536-dim space.

**Q: How does masked self-attention work in SQL generation?**

A: When generating SQL, the model predicts one token at a time. At each step, the current token can attend to (look at) ALL previous tokens, but NOT future tokens — this is the "mask." So when generating "customer_id" after "SELECT", the model attends to: the full prompt (schemas, question, rules) + "SELECT". It knows we're selecting columns, and based on the schema context, it picks "customer_id" as the most probable next token. This autoregressive, left-to-right generation is why prompt quality matters so much — the schemas in the prompt are the ONLY source of truth.

**Q: Explain cosine similarity mathematically and intuitively.**

A: **Math**: `sim(A, B) = dot(A, B) / (||A|| × ||B||)` where dot product = sum of element-wise multiplication, and ||A|| = sqrt of sum of squares (L2 norm). Result ranges from -1 to 1. **Intuitively**: Imagine two arrows in 1536-dimensional space. Cosine measures the ANGLE between them. If they point the same direction (small angle) → high similarity. If they're perpendicular (90°) → zero similarity. **Why not Euclidean**: Euclidean measures absolute distance — affected by vector LENGTH. A 100-word document about "fraud" and a 10-word document about "fraud" have different lengths but the same direction. Cosine correctly says they're similar; Euclidean would say they're far apart.

**Q: What prompting techniques do you use and when?**

A: 
- **Zero-shot**: Simple SQL queries. Just schema + question + rules. Works when the query pattern is straightforward.
- **Few-shot**: Complex joins or banking-specific patterns. Include 2-3 examples of question → SQL. The examples teach the model our conventions (BigQuery syntax, partition filters, join patterns).
- **Chain-of-thought**: Multi-table analysis. "Think step by step: what data do I need? Which tables? How to join?" Forces the model to reason before generating SQL.
- **ReAct**: Agent investigation. "THOUGHT: what to investigate → ACTION: which tool → OBSERVATION: result." The structured format keeps the agent on track and produces parseable JSON actions.

---

### System Design

**Q: How would you scale this system for 100x more schemas?**

A: At 100K schemas:
1. **Replace custom vector store with FAISS** — Facebook's library for efficient similarity search. Uses approximate nearest neighbor (ANN) — trades minor accuracy for 100x speed.
2. **Hierarchical retrieval** — First retrieve relevant domains (fraud, compliance, transactions), then search within that domain. Two-stage retrieval.
3. **Schema caching** — Cache popular schema embeddings in Redis. 80% of queries hit the same 20% of schemas.
4. **Embedding pre-computation** — Compute and store schema embeddings at indexing time. Only re-compute when schema changes.

**Q: How would you handle multiple LLM providers failing simultaneously?**

A: Defense in depth:
1. **Primary**: OpenAI (lowest latency)
2. **Fallback 1**: Anthropic (auto-switch on 5xx errors)
3. **Fallback 2**: Self-hosted model (runs on our infrastructure)
4. **Circuit breaker**: After 3 failures in 60 seconds, switch to fallback
5. **Degraded mode**: If all LLMs fail, return raw SQL results without natural language summary

---

### Behavioral / STAR

**Q: Why did you build this project?**

A: I wanted to go beyond theoretical AI knowledge into production implementation. Every data engineer talks about "AI integration" but few have actually built the vector store, designed the embedding pipeline, implemented the ReAct agent loop, and handled the edge cases (SQL injection, cost control, provider failover). This project demonstrates that I can design and build the entire AI stack — not just use a library.

**Q: What was the hardest design decision?**

A: **Situation**: Choosing between fine-tuning an LLM on banking data vs RAG. **Task**: Maximize SQL generation accuracy. **Action**: I prototyped both. Fine-tuning required 500+ labeled question-SQL pairs (didn't have), GPU compute ($500+/training run), and goes stale when schemas change. RAG uses actual schemas in real-time (always current), costs only per API call, and adapts instantly when tables are added/changed. **Result**: RAG achieved ~90% accuracy at a fraction of the cost. The deciding factor: schemas change monthly — fine-tuning would need monthly retraining.

---

## 7. Key Numbers to Remember

| Metric | Value | Context |
|--------|-------|---------|
| Pipeline throughput | 5M+ records/month | Across banking sources |
| Query performance improvement | +25% | Partitioning + clustering |
| Bad-record rate | <1% | 5-check quality framework |
| NL query accuracy (RAG) | ~90% | Schema retrieval + few-shot |
| Agent investigation time | Minutes | vs hours manual |
| CI/CD deployment | <10 min | PR to production |
| Test coverage | 92% | Unit + integration |
| Vector search latency | <10ms | 1000 schemas, 1536 dims |
| Embedding dimensions | 1536 | Contextual, transformer-based |
| Max agent steps | 10 | $0.50 cost cap per investigation |
