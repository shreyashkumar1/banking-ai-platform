"""Tests for the production Embedding Engine."""

import numpy as np
import pytest
from src.ai.embedding_engine import (
    EmbeddingEngine, EmbeddingResult, CircuitBreaker,
    MockProvider, OpenAIProvider, PROVIDERS, EngineMetrics,
)


class TestCircuitBreaker:

    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb.can_execute() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() is False

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failures == 0
        assert cb.state == "CLOSED"

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        assert cb.state == "OPEN"
        # With 0 cooldown, should immediately go to half-open
        assert cb.can_execute() is True
        assert cb.state == "HALF_OPEN"


class TestMockProvider:

    def test_deterministic_embeddings(self):
        provider = MockProvider(dims=1536)
        v1 = provider.embed(["test"])[0]
        v2 = provider.embed(["test"])[0]
        np.testing.assert_array_equal(v1, v2)  # Same text → same vector

    def test_different_texts_different_vectors(self):
        provider = MockProvider(dims=1536)
        v1 = provider.embed(["fraud"])[0]
        v2 = provider.embed(["weather"])[0]
        assert not np.array_equal(v1, v2)

    def test_normalized_output(self):
        provider = MockProvider(dims=1536)
        vec = np.array(provider.embed(["test"])[0])
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5  # Unit vector

    def test_correct_dimensions(self):
        provider = MockProvider(dims=768)
        vec = provider.embed(["test"])[0]
        assert len(vec) == 768


class TestEmbeddingEngine:

    def setup_method(self):
        self.engine = EmbeddingEngine(provider="mock", cache_size=100)

    def test_embed_text_returns_result(self):
        result = self.engine.embed_text("fraud detection in banking")
        assert isinstance(result, EmbeddingResult)
        assert result.dimensions == 1536
        assert len(result.vector) == 1536
        assert result.cached is False

    def test_embed_text_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            self.engine.embed_text("")

    def test_embed_text_whitespace_raises(self):
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            self.engine.embed_text("   ")

    def test_caching_returns_cached_result(self):
        r1 = self.engine.embed_text("test query")
        r2 = self.engine.embed_text("test query")
        assert r2.cached is True
        np.testing.assert_array_equal(r1.vector, r2.vector)

    def test_cache_hit_rate_tracking(self):
        self.engine.embed_text("query 1")
        self.engine.embed_text("query 1")  # Cache hit
        self.engine.embed_text("query 2")  # Cache miss
        assert self.engine.metrics.cache_hits == 1
        assert self.engine.metrics.cache_misses == 2

    def test_cache_eviction_at_max_size(self):
        engine = EmbeddingEngine(provider="mock", cache_size=3)
        for i in range(5):
            engine.embed_text(f"query {i}")
        assert len(engine._cache) == 3  # Max 3 entries

    def test_embed_batch_returns_all(self):
        texts = ["fraud", "banking", "compliance"]
        results = self.engine.embed_batch(texts)
        assert len(results) == 3
        for r in results:
            assert r.dimensions == 1536

    def test_embed_batch_uses_cache(self):
        self.engine.embed_text("cached query")
        results = self.engine.embed_batch(["cached query", "new query"])
        assert results[0].cached is True
        assert results[1].cached is False

    def test_cosine_similarity_identical(self):
        vec = np.array([1.0, 0.0, 0.0])
        assert abs(EmbeddingEngine.cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(EmbeddingEngine.cosine_similarity(a, b)) < 1e-6

    def test_find_most_similar_ordering(self):
        query = self.engine.embed_text("fraud detection").vector
        candidates = [
            self.engine.embed_text("fraud alerts and detection"),
            self.engine.embed_text("weather forecast tomorrow"),
            self.engine.embed_text("banking fraud patterns"),
        ]
        results = self.engine.find_most_similar(query, candidates, top_k=2)
        assert len(results) == 2
        assert results[0][0] >= results[1][0]  # Descending similarity

    def test_metrics_export(self):
        self.engine.embed_text("test")
        metrics = self.engine.get_metrics()
        assert "total_requests" in metrics
        assert "cache_hit_rate" in metrics
        assert "circuit_breaker_state" in metrics

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            EmbeddingEngine(provider="nonexistent")

    def test_vector_normalization(self):
        result = self.engine.embed_text("test normalization")
        norm = np.linalg.norm(result.vector)
        assert abs(norm - 1.0) < 1e-5  # Unit vector


class TestEngineMetrics:

    def test_cache_hit_rate_zero_division(self):
        m = EngineMetrics()
        assert m.cache_hit_rate == 0.0

    def test_cache_hit_rate_calculation(self):
        m = EngineMetrics(cache_hits=3, cache_misses=7)
        assert abs(m.cache_hit_rate - 0.3) < 1e-6

    def test_avg_latency(self):
        m = EngineMetrics(total_requests=10, total_latency_ms=500)
        assert abs(m.avg_latency_ms - 50.0) < 1e-6

    def test_error_rate(self):
        m = EngineMetrics(total_requests=100, api_errors=5)
        assert abs(m.error_rate - 0.05) < 1e-6
