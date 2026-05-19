"""
Vector Store — Semantic search over banking data schemas.

WHY: When a user asks "show me high-value suspicious transactions", we need to find
which BigQuery tables/columns are relevant. Keyword search fails because the user
says "suspicious" but the column is named "fraud_score". Vector search matches by
MEANING using cosine similarity between embedding vectors.

WHERE USED:
- RAG Pipeline: First step — find relevant schemas before generating SQL
- Document Search: Find relevant compliance docs for audit questions
- Similar Pattern Matching: Find transactions similar to known fraud patterns

HOW IT WORKS:
1. INDEXING (one-time): Convert all table/column descriptions → embedding vectors
2. SEARCH (per query): Convert user question → embedding vector
3. MATCH: Find closest vectors using cosine similarity (1 - cos θ)
4. RETURN: Top-K most relevant schemas with similarity scores
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json
import logging

from src.ai.embedding_engine import EmbeddingEngine, EmbeddingResult

logger = logging.getLogger(__name__)


@dataclass
class SchemaEntry:
    """Represents a BigQuery table/column indexed in the vector store."""
    table_name: str
    description: str
    columns: dict[str, str]
    sample_queries: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """Result from a vector search query."""
    schema: SchemaEntry
    similarity_score: float
    rank: int


class VectorStore:
    """In-memory vector store for schema semantic search.
    
    Architecture Decision — Why custom (not Pinecone/Weaviate/ChromaDB)?
    - Banking data schemas are small (<1000 entries) — full vector DB is overkill
    - No external dependency = no data leaving the network (compliance requirement)
    - Brute-force cosine similarity on 1000 vectors takes <10ms
    - If schema catalog grows >10K entries, migrate to managed vector DB
    
    How Cosine Similarity Works Here:
    - User question: "high value transactions" → vector A (1536 dims)
    - Schema "fact_transactions: Transaction amounts and types" → vector B
    - cos(θ) = A·B / (||A||×||B||)
    - If θ is small (vectors point same direction) → high similarity → MATCH
    - If θ is large (vectors point different directions) → low similarity → NO MATCH
    """

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.engine = embedding_engine or EmbeddingEngine()
        self.entries: list[SchemaEntry] = []
        self._indexed = False

    def index_schemas(self, schemas: dict[str, dict]):
        """Index all BigQuery table schemas as embedding vectors.
        
        This is the "training" phase — like how Word2Vec is trained on Wikipedia,
        we pre-compute embeddings for all our schemas so search is instant.
        
        Args:
            schemas: Dict of table_name → {description, columns, sample_queries}
        """
        logger.info(f"Indexing {len(schemas)} schemas into vector store...")
        
        texts_to_embed = []
        entries = []

        for table_name, info in schemas.items():
            # Create rich text representation for embedding
            # More context in the text = better embedding quality
            text = f"Table: {table_name}. {info['description']}. "
            text += "Columns: " + ", ".join(
                f"{col} ({desc})" for col, desc in info.get('columns', {}).items()
            )
            if info.get('sample_queries'):
                text += " Example queries: " + "; ".join(info['sample_queries'][:3])

            texts_to_embed.append(text)
            entries.append(SchemaEntry(
                table_name=table_name,
                description=info['description'],
                columns=info.get('columns', {}),
                sample_queries=info.get('sample_queries', [])
            ))

        # Batch embed all schemas
        results = self.engine.embed_batch(texts_to_embed)

        for entry, result in zip(entries, results):
            entry.embedding = result.vector

        self.entries = entries
        self._indexed = True
        logger.info(f"Indexed {len(entries)} schemas. Vector store ready.")

    def search(self, query: str, top_k: int = 3, min_score: float = 0.3) -> list[SearchResult]:
        """Semantic search: find schemas most relevant to a natural language question.
        
        This is the core of how RAG works:
        1. Embed the user's question → 1536-dim vector
        2. Compare against ALL indexed schema vectors using cosine similarity
        3. Return top-K matches above minimum similarity threshold
        
        Why min_score=0.3?
        - Below 0.3 cosine similarity, results are essentially random
        - Banking queries need precision — false positives waste LLM context tokens
        - Better to return fewer, highly relevant schemas than many weak matches
        
        Args:
            query: Natural language question from user
            top_k: Number of results to return
            min_score: Minimum cosine similarity threshold
        """
        if not self._indexed:
            raise RuntimeError("Vector store not indexed. Call index_schemas() first.")

        # Step 1: Embed the question
        query_embedding = self.engine.embed_text(query)

        # Step 2: Calculate cosine similarity against all indexed schemas
        results = []
        for entry in self.entries:
            score = self.engine.cosine_similarity(
                query_embedding.vector, entry.embedding
            )
            if score >= min_score:
                results.append(SearchResult(
                    schema=entry, similarity_score=score, rank=0
                ))

        # Step 3: Sort by similarity (descending) and assign ranks
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1

        logger.info(
            f"Search '{query[:50]}...' returned {len(results[:top_k])} results "
            f"(top score: {results[0].similarity_score:.3f})" if results else
            f"Search '{query[:50]}...' returned 0 results"
        )

        return results[:top_k]

    def get_context_for_prompt(self, query: str, top_k: int = 3) -> str:
        """Get formatted context string for LLM prompt injection.
        
        This is the 'Retrieval' in RAG — the retrieved context gets injected
        into the LLM prompt so the model can generate accurate SQL.
        
        Returns formatted string like:
        Table: analytics.fact_transactions
        Description: Transaction events with amounts
        Columns: customer_id (unique ID), amount (INR), ...
        """
        results = self.search(query, top_k=top_k)
        
        context_parts = []
        for r in results:
            part = f"Table: {r.schema.table_name}\n"
            part += f"Description: {r.schema.description}\n"
            part += "Columns:\n"
            for col, desc in r.schema.columns.items():
                part += f"  - {col}: {desc}\n"
            if r.schema.sample_queries:
                part += "Example queries:\n"
                for sq in r.schema.sample_queries[:2]:
                    part += f"  - {sq}\n"
            context_parts.append(part)

        return "\n---\n".join(context_parts)


# ── Banking Schema Registry ──
BANKING_SCHEMAS = {
    "analytics.fact_transactions": {
        "description": "All banking transactions including transfers, payments, withdrawals, and deposits",
        "columns": {
            "transaction_id": "Unique transaction identifier",
            "account_id": "Source account ID (join key to dim_account)",
            "customer_id": "Customer ID (join key to dim_customer)",
            "transaction_type": "TRANSFER, PAYMENT, WITHDRAWAL, DEPOSIT, LOAN_DISBURSEMENT",
            "amount": "Transaction amount in INR",
            "currency": "Currency code (INR, USD, EUR)",
            "status": "COMPLETED, PENDING, FAILED, REVERSED",
            "channel": "MOBILE, WEB, ATM, BRANCH, UPI",
            "fraud_score": "ML-generated fraud probability (0.0 to 1.0)",
            "created_date": "Transaction date (partition key)",
            "created_timestamp": "Full timestamp with timezone",
        },
        "sample_queries": [
            "SELECT customer_id, SUM(amount) as total FROM fact_transactions WHERE created_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) GROUP BY 1",
            "SELECT transaction_type, COUNT(*) as cnt, AVG(amount) as avg_amt FROM fact_transactions WHERE created_date = CURRENT_DATE() GROUP BY 1",
        ]
    },
    "analytics.fact_fraud_alerts": {
        "description": "Fraud detection alerts generated by ML models and rule engines",
        "columns": {
            "alert_id": "Unique alert identifier",
            "transaction_id": "Related transaction ID",
            "customer_id": "Flagged customer ID",
            "alert_type": "VELOCITY, AMOUNT_ANOMALY, GEO_ANOMALY, PATTERN_MATCH",
            "risk_level": "HIGH, MEDIUM, LOW",
            "fraud_score": "ML model confidence (0.0 to 1.0)",
            "status": "OPEN, INVESTIGATING, CONFIRMED_FRAUD, FALSE_POSITIVE",
            "created_date": "Alert date (partition key)",
        },
        "sample_queries": [
            "SELECT risk_level, COUNT(*) FROM fact_fraud_alerts WHERE created_date = CURRENT_DATE() GROUP BY 1",
        ]
    },
    "analytics.dim_customer": {
        "description": "Customer master data with KYC information and segmentation",
        "columns": {
            "customer_id": "Unique customer identifier (join key)",
            "customer_name": "Full name (PII — column-level security applied)",
            "customer_segment": "RETAIL, HNI, CORPORATE, SME",
            "kyc_status": "VERIFIED, PENDING, EXPIRED, REJECTED",
            "risk_category": "LOW, MEDIUM, HIGH (AML risk rating)",
            "onboarding_date": "Account opening date",
            "region": "Geographic region",
        }
    },
    "analytics.dim_account": {
        "description": "Bank account details including type and status",
        "columns": {
            "account_id": "Unique account identifier",
            "customer_id": "Owner customer ID",
            "account_type": "SAVINGS, CURRENT, LOAN, FIXED_DEPOSIT, DEMAT",
            "balance": "Current account balance in INR",
            "status": "ACTIVE, DORMANT, CLOSED, FROZEN",
            "branch_code": "Branch identifier",
        }
    },
}
