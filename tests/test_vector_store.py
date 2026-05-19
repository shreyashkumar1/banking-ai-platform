"""Tests for the Vector Store."""

import pytest
from unittest.mock import MagicMock
from src.ai.vector_store import VectorStore, BANKING_SCHEMAS, SchemaEntry, SearchResult
from src.ai.embedding_engine import EmbeddingEngine


class TestVectorStore:

    def setup_method(self):
        self.engine = EmbeddingEngine(provider="mock")
        self.store = VectorStore(self.engine)

    def test_index_schemas_creates_entries(self):
        self.store.index_schemas(BANKING_SCHEMAS)
        assert len(self.store.entries) == len(BANKING_SCHEMAS)
        assert self.store._indexed is True

    def test_index_schemas_all_have_embeddings(self):
        self.store.index_schemas(BANKING_SCHEMAS)
        for entry in self.store.entries:
            assert entry.embedding is not None
            assert len(entry.embedding) == 1536

    def test_search_requires_indexing(self):
        with pytest.raises(RuntimeError, match="not indexed"):
            self.store.search("fraud alerts")

    def test_search_returns_results(self):
        self.store.index_schemas(BANKING_SCHEMAS)
        results = self.store.search("fraud alerts", top_k=2)
        assert len(results) <= 2
        for r in results:
            assert isinstance(r, SearchResult)
            assert r.rank > 0
            assert 0 <= r.similarity_score <= 1

    def test_search_results_ordered_by_similarity(self):
        self.store.index_schemas(BANKING_SCHEMAS)
        results = self.store.search("transaction amount", top_k=3)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i + 1].similarity_score

    def test_search_min_score_filter(self):
        self.store.index_schemas(BANKING_SCHEMAS)
        results = self.store.search("random unrelated query xyz", top_k=10, min_score=0.99)
        # With high min_score, might get 0 results
        for r in results:
            assert r.similarity_score >= 0.99

    def test_get_context_for_prompt(self):
        self.store.index_schemas(BANKING_SCHEMAS)
        # Force search to return something so we can test the formatting
        self.store.search = lambda q, top_k: [
            __import__('src.ai.vector_store', fromlist=['SearchResult']).SearchResult(
                schema=self.store.entries[0], similarity_score=0.9, rank=1
            )
        ]
        context = self.store.get_context_for_prompt("fraud alerts", top_k=2)
        assert isinstance(context, str)
        assert "Table:" in context
        assert "Columns:" in context

    def test_schema_entry_has_all_fields(self):
        self.store.index_schemas(BANKING_SCHEMAS)
        for entry in self.store.entries:
            assert entry.table_name is not None
            assert entry.description is not None
            assert isinstance(entry.columns, dict)

    def test_banking_schemas_well_defined(self):
        """Verify our banking schema registry has required fields."""
        for table_name, info in BANKING_SCHEMAS.items():
            assert "description" in info
            assert "columns" in info
            assert len(info["columns"]) > 0
            assert "." in table_name  # Should be dataset.table format
