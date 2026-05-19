"""
demo.py — Runnable demo of the Banking AI Platform (no GCP credentials needed)

Shows the full AI stack working end-to-end with mock data:
1. Embedding Engine  — text → 1536-dim vectors with caching
2. Vector Store      — schema semantic search
3. RAG Query Engine  — natural language → SQL pipeline
4. Investigation Agent — autonomous multi-step ReAct loop

Run: python demo.py
"""

import json
import time
import sys

# ── Colour output (works on Mac/Linux) ──
GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def header(text: str):
    print(f"\n{BOLD}{BLUE}{'═' * 60}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'═' * 60}{RESET}\n")

def step(label: str, value: str = ""):
    print(f"  {GREEN}▶{RESET} {BOLD}{label}{RESET}", end="")
    if value:
        print(f"  {DIM}{value}{RESET}", end="")
    print()

def result(text: str):
    for line in text.split("\n"):
        print(f"    {YELLOW}{line}{RESET}")

def error(text: str):
    print(f"  {RED}✗ {text}{RESET}")


# ══════════════════════════════════════════════════════════════════
# DEMO 1 — Embedding Engine
# ══════════════════════════════════════════════════════════════════

def demo_embedding():
    header("DEMO 1: Embedding Engine")

    from src.ai.embedding_engine import EmbeddingEngine
    import numpy as np

    engine = EmbeddingEngine(provider="mock", cache_size=1000)

    # Show embedding
    step("Embedding 'high value suspicious transactions'...")
    r1 = engine.embed_text("high value suspicious transactions")
    print(f"\n    Dimensions  : {r1.dimensions}")
    print(f"    Vector[:5]  : {r1.vector[:5].round(4)}")
    print(f"    L2 norm     : {np.linalg.norm(r1.vector):.6f}  (unit vector ✓)")
    print(f"    Cached      : {r1.cached}")
    print(f"    Latency     : {r1.latency_ms:.1f}ms")

    # Show caching
    step("\nEmbedding same text again (cache hit)...")
    r1_cached = engine.embed_text("high value suspicious transactions")
    print(f"\n    Cached      : {r1_cached.cached}  ✓")
    print(f"    Latency     : {r1_cached.latency_ms:.2f}ms  (near-zero)")

    # Show semantic proximity
    step("\nCosine similarity test...")
    r2 = engine.embed_text("fraud detection alert")
    r3 = engine.embed_text("monsoon weather forecast")

    sim_fraud = EmbeddingEngine.cosine_similarity(r1.vector, r2.vector)
    sim_weather = EmbeddingEngine.cosine_similarity(r1.vector, r3.vector)

    print(f"\n    'suspicious transactions' ↔ 'fraud detection alert'  : {sim_fraud:.4f}")
    print(f"    'suspicious transactions' ↔ 'monsoon weather forecast': {sim_weather:.4f}")
    print(f"\n    {GREEN}Result: fraud query is {abs(sim_fraud - sim_weather) * 100:.0f}+ points more similar to suspicious transactions{RESET}")

    # Metrics
    step("\nEngine metrics...")
    metrics = engine.get_metrics()
    result(json.dumps(metrics, indent=4))


# ══════════════════════════════════════════════════════════════════
# DEMO 2 — Vector Store Schema Search
# ══════════════════════════════════════════════════════════════════

def demo_vector_store():
    header("DEMO 2: Vector Store — Schema Semantic Search")

    from src.ai.embedding_engine import EmbeddingEngine
    from src.ai.vector_store import VectorStore, BANKING_SCHEMAS

    engine = EmbeddingEngine(provider="mock")
    store = VectorStore(engine)

    step(f"Indexing {len(BANKING_SCHEMAS)} banking schemas...")
    t0 = time.time()
    store.index_schemas(BANKING_SCHEMAS)
    elapsed = (time.time() - t0) * 1000
    print(f"  {DIM}→ {elapsed:.1f}ms{RESET}")

    # Run 4 different natural language queries
    queries = [
        "show me fraud alerts for high risk customers",
        "what is the customer account balance",
        "UPI payment failures today",
        "KYC verification status",
    ]

    for q in queries:
        step(f"\nQuery: \"{q}\"")
        results = store.search(q, top_k=2)
        for r in results:
            print(f"    {GREEN}[{r.similarity_score:.3f}]{RESET} {r.schema.table_name}  —  {r.schema.description[:60]}...")


# ══════════════════════════════════════════════════════════════════
# DEMO 3 — SQL Validator (Security)
# ══════════════════════════════════════════════════════════════════

def demo_sql_validator():
    header("DEMO 3: SQL Injection Prevention")

    from src.ai.rag_query_engine import SQLValidator

    ALLOWED = {
        "analytics.fact_transactions",
        "analytics.fact_fraud_alerts",
        "analytics.dim_customer",
    }

    test_cases = [
        ("SELECT customer_id, SUM(amount) FROM `analytics.fact_transactions` WHERE created_date = CURRENT_DATE() GROUP BY 1", "Valid SELECT"),
        ("DROP TABLE analytics.fact_transactions", "DROP attack"),
        ("SELECT * FROM t; DELETE FROM t", "Semicolon injection"),
        ("UPDATE analytics.fact_transactions SET amount = 0", "UPDATE attack"),
        ("WITH cte AS (SELECT * FROM `analytics.fact_transactions`) SELECT * FROM cte", "Valid CTE"),
        ("INSERT INTO analytics.fact_transactions VALUES ('x', 'y')", "INSERT attack"),
    ]

    for sql, label in test_cases:
        valid, error_msg = SQLValidator.validate(sql, ALLOWED)
        icon = f"{GREEN}✓ ALLOW{RESET}" if valid else f"{RED}✗ BLOCK{RESET}"
        print(f"  {icon}  [{label}]")
        if not valid:
            print(f"         {DIM}→ {error_msg}{RESET}")


# ══════════════════════════════════════════════════════════════════
# DEMO 4 — Query Cache
# ══════════════════════════════════════════════════════════════════

def demo_query_cache():
    header("DEMO 4: Query Result Caching")

    from src.ai.rag_query_engine import QueryCache, QueryResult

    cache = QueryCache(ttl_seconds=60, max_entries=100)

    # Simulate putting a result
    dummy_result = QueryResult(
        question="How many transactions today?",
        sql="SELECT COUNT(*) FROM analytics.fact_transactions WHERE created_date = CURRENT_DATE()",
        raw_results=[{"total": 15432}],
        answer="There were 15,432 transactions today.",
        schemas_used=["analytics.fact_transactions"],
        latency_ms=420,
    )

    step("Storing query result...")
    cache.put("How many transactions today?", dummy_result)

    step("Cache lookup (same question)...")
    hit = cache.get("How many transactions today?")
    print(f"\n    Cached     : {hit.cached}")
    print(f"    Answer     : {hit.answer}")
    print(f"    Hit rate   : {cache.hit_rate:.0%}")

    step("\nCache lookup (unseen question)...")
    miss = cache.get("What are the fraud stats?")
    print(f"\n    Result     : {miss}  (miss — will call LLM + BigQuery)")


# ══════════════════════════════════════════════════════════════════
# DEMO 5 — Investigation Agent (Full ReAct Loop)
# ══════════════════════════════════════════════════════════════════

def demo_agent():
    header("DEMO 5: Autonomous Investigation Agent (ReAct Loop)")

    from src.ai.agent import BankingInvestigationAgent

    # Mock BQ client — won't actually query GCP
    class MockBQ:
        def query(self, sql):
            class Result:
                def result(self):
                    return [{"risk_level": "HIGH", "cnt": 47, "avg_score": 0.89}]
            return Result()
        def get_table(self, _):
            raise Exception("Table not found (mock)")

    # Patch BQ with mock
    import unittest.mock as mock
    with mock.patch("google.cloud.bigquery.Client") as MockClient:
        MockClient.return_value = MockBQ()

        agent = BankingInvestigationAgent(
            project_id="banking-ai-demo",
            max_steps=10,
        )

    print(f"  {DIM}Objective: Investigate spike in HIGH-risk fraud alerts today{RESET}\n")

    t0 = time.time()
    # Run with mock LLM (built-in, simulates real investigation)
    result_obj = agent.investigate(
        "Investigate the spike in HIGH-risk fraud alerts today — identify affected customers and recommend action"
    )

    elapsed = time.time() - t0

    print(f"\n  {BOLD}Investigation Results:{RESET}")
    print(f"  {'─' * 56}")
    print(f"  Steps completed   : {len(result_obj.steps)}")
    print(f"  SQL queries run   : {result_obj.total_queries}")
    print(f"  Duration          : {elapsed:.2f}s")
    print(f"  Cost estimate     : ${result_obj.cost_estimate_usd:.2f}")
    print(f"  Completed         : {result_obj.completed}")

    print(f"\n  {BOLD}Step-by-step reasoning:{RESET}")
    for s in result_obj.steps:
        icon = f"{GREEN}✓{RESET}" if s.success else f"{RED}✗{RESET}"
        print(f"\n  {icon} Step {s.step_number}: {BOLD}{s.tool}{RESET}")
        print(f"    Thought   : {s.thought[:80]}...")
        print(f"    Reasoning : {s.reasoning[:70]}...")
        print(f"    Duration  : {s.duration_ms:.1f}ms")

    print(f"\n  {BOLD}{GREEN}Summary:{RESET}")
    for line in result_obj.summary.split(". "):
        if line.strip():
            print(f"    • {line.strip()}")

    print(f"\n  {BOLD}Root Cause:{RESET}")
    print(f"    {result_obj.root_cause}")

    print(f"\n  {BOLD}Recommendation:{RESET}")
    print(f"    {result_obj.recommendation}")

    # Audit trail
    step("\nAudit trail (compliance export)...")
    audit = agent.get_audit_trail(result_obj)
    print(f"\n    Fields: {list(audit.keys())}")
    print(f"    Tool usage: {audit['tool_usage']}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  Banking AI Platform — Live Demo")
    print(f"  Provider-agnostic RAG + Agentic AI for Banking")
    print(f"{'═' * 60}{RESET}")

    demos = {
        "1": ("Embedding Engine (caching, normalization, metrics)", demo_embedding),
        "2": ("Vector Store (schema semantic search)", demo_vector_store),
        "3": ("SQL Validator (injection prevention)", demo_sql_validator),
        "4": ("Query Cache (cost optimization)", demo_query_cache),
        "5": ("Investigation Agent (full ReAct loop)", demo_agent),
    }

    # If arg given, run specific demo
    if len(sys.argv) > 1 and sys.argv[1] in demos:
        name, fn = demos[sys.argv[1]]
        fn()
    else:
        # Run all
        for key, (name, fn) in demos.items():
            try:
                fn()
            except Exception as e:
                error(f"Demo {key} failed: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{BOLD}{GREEN}{'═' * 60}")
    print(f"  All demos complete")
    print(f"{'═' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
