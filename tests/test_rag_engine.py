"""Tests for SQL Validator and RAG Engine safety features."""

import pytest
from src.ai.rag_query_engine import SQLValidator, QueryCache


class TestSQLValidator:
    """SQL injection prevention is critical in banking."""

    ALLOWED_TABLES = {
        "analytics.fact_transactions",
        "analytics.dim_customer",
        "analytics.fact_fraud_alerts",
    }

    def test_valid_select(self):
        sql = "SELECT customer_id, COUNT(*) FROM `analytics.fact_transactions` GROUP BY 1"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is True

    def test_valid_cte(self):
        sql = "WITH daily AS (SELECT * FROM `analytics.fact_transactions`) SELECT * FROM daily"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is True

    def test_blocks_drop_table(self):
        sql = "DROP TABLE analytics.fact_transactions"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is False
        assert "Only SELECT" in err

    def test_blocks_delete(self):
        sql = "DELETE FROM analytics.fact_transactions WHERE 1=1"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is False

    def test_blocks_update(self):
        sql = "UPDATE analytics.fact_transactions SET amount = 0"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is False

    def test_blocks_injection_via_semicolon(self):
        sql = "SELECT * FROM analytics.fact_transactions; SELECT * FROM analytics.dim_customer"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is False
        assert "Multiple SQL statements" in err

    def test_blocks_insert(self):
        sql = "INSERT INTO analytics.fact_transactions VALUES ('hack', 1)"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is False

    def test_allows_columns_with_blocked_substrings(self):
        # "update_date" contains "UPDATE" but shouldn't be blocked
        sql = "SELECT update_date, delete_flag FROM `analytics.fact_transactions`"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is True  # Words within column names are NOT blocked

    def test_blocks_alter(self):
        sql = "ALTER TABLE analytics.fact_transactions ADD COLUMN hack STRING"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is False

    def test_blocks_grant(self):
        sql = "GRANT ALL ON analytics.fact_transactions TO 'hacker'"
        valid, err = SQLValidator.validate(sql, self.ALLOWED_TABLES)
        assert valid is False


class TestQueryCache:

    def test_cache_miss(self):
        cache = QueryCache(ttl_seconds=60)
        assert cache.get("new question") is None
        assert cache.misses == 1

    def test_cache_hit(self):
        from src.ai.rag_query_engine import QueryResult
        cache = QueryCache(ttl_seconds=60)
        result = QueryResult(
            question="test", sql="SELECT 1", raw_results=[],
            answer="test answer", schemas_used=[], latency_ms=10
        )
        cache.put("test question", result)
        cached = cache.get("test question")
        assert cached is not None
        assert cached.cached is True
        assert cache.hits == 1

    def test_cache_expiry(self):
        cache = QueryCache(ttl_seconds=0)  # Immediate expiry
        from src.ai.rag_query_engine import QueryResult
        result = QueryResult(
            question="test", sql="SELECT 1", raw_results=[],
            answer="test", schemas_used=[], latency_ms=10
        )
        cache.put("test", result)
        import time
        time.sleep(0.01)
        assert cache.get("test") is None  # Expired

    def test_cache_eviction_at_max(self):
        cache = QueryCache(ttl_seconds=300, max_entries=2)
        from src.ai.rag_query_engine import QueryResult
        for i in range(5):
            result = QueryResult(
                question=f"q{i}", sql="SELECT 1", raw_results=[],
                answer=f"a{i}", schemas_used=[], latency_ms=10
            )
            cache.put(f"question {i}", result)
        assert len(cache._cache) == 2

    def test_hit_rate(self):
        cache = QueryCache()
        cache.hits = 3
        cache.misses = 7
        assert abs(cache.hit_rate - 0.3) < 1e-6
