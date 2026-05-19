"""BigQuery Client Helper — Centralized BQ operations."""

from google.cloud import bigquery
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BQClient:
    """Wrapper around BigQuery client with common banking operations."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def query(self, sql: str, max_results: int = 1000) -> list[dict]:
        """Execute a SQL query and return results as list of dicts."""
        try:
            job = self.client.query(sql)
            return [dict(row) for row in job.result()][:max_results]
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            raise

    def get_table_schema(self, table: str) -> dict[str, str]:
        """Get schema of a table as {column_name: field_type}."""
        t = self.client.get_table(f"{self.project_id}.{table}")
        return {f.name: f.field_type for f in t.schema}

    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        try:
            self.client.get_table(f"{self.project_id}.{table}")
            return True
        except Exception:
            return False

    def get_row_count(self, table: str, date_filter: Optional[str] = None) -> int:
        """Get row count, optionally filtered by date."""
        where = f"WHERE created_date = '{date_filter}'" if date_filter else ""
        sql = f"SELECT COUNT(*) as cnt FROM `{self.project_id}.{table}` {where}"
        result = self.query(sql)
        return result[0]["cnt"] if result else 0
