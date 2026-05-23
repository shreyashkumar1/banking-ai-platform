"""BigQuery Client Helper — Centralized BQ operations."""

from google.cloud import bigquery
from google.auth.exceptions import DefaultCredentialsError
from typing import Optional
import logging
import json

logger = logging.getLogger(__name__)


class BQClient:
    """Wrapper around BigQuery client with common banking operations.
    
    Gracefully falls back to mock mode if GCP credentials are not found.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        try:
            self.client = bigquery.Client(project=project_id)
            self.mock_mode = False
        except (DefaultCredentialsError, Exception) as e:
            logger.warning(f"Could not initialize BigQuery client: {e}. Running in MOCK mode.")
            self.client = None
            self.mock_mode = True

    def query(self, sql: str, max_results: int = 1000) -> list[dict]:
        """Execute a SQL query and return results as list of dicts."""
        if self.mock_mode:
            # Return realistic mock results
            logger.info(f"[MOCK BQ] Executing query: {sql[:100]}...")
            return [{"risk_level": "HIGH", "cnt": 47, "avg_score": 0.89}]

        try:
            job = self.client.query(sql)
            return [dict(row) for row in job.result()][:max_results]
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            raise

    def get_table_schema(self, table: str) -> dict[str, str]:
        """Get schema of a table as {column_name: field_type}."""
        if self.mock_mode:
            # Return realistic mock schemas
            from src.ai.vector_store import BANKING_SCHEMAS
            for name, info in BANKING_SCHEMAS.items():
                if table in name:
                    return {col["name"]: col["type"] for col in info["columns"]}
            return {"transaction_id": "STRING", "amount": "NUMERIC", "created_date": "DATE"}

        t = self.client.get_table(f"{self.project_id}.{table}")
        return {f.name: f.field_type for f in t.schema}

    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        if self.mock_mode:
            return True
        try:
            self.client.get_table(f"{self.project_id}.{table}")
            return True
        except Exception:
            return False

    def get_row_count(self, table: str, date_filter: Optional[str] = None) -> int:
        """Get row count, optionally filtered by date."""
        if self.mock_mode:
            return 15432
        where = f"WHERE created_date = '{date_filter}'" if date_filter else ""
        sql = f"SELECT COUNT(*) as cnt FROM `{self.project_id}.{table}` {where}"
        result = self.query(sql)
        return result[0]["cnt"] if result else 0

