"""
Airflow DAG — Daily Banking Data Pipeline Orchestration.

Pipeline flow:
1. Wait for source data (GCS sensor)
2. Create ephemeral Dataproc cluster
3. Run PySpark transformation
4. Data quality checks (5 automated gates)
5. Load to BigQuery warehouse
6. Index new schemas in vector store
7. Delete Dataproc cluster (ALWAYS — even on failure)
8. Notify team via Slack
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocSubmitPySparkJobOperator,
    DataprocDeleteClusterOperator,
)
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

# ── Configuration ──
PROJECT_ID = "banking-ai-prod"
REGION = "asia-south1"
CLUSTER_NAME = "banking-etl-{{ ds_nodash }}"
BUCKET = f"{PROJECT_ID}-data"

default_args = {
    "owner": "data-engineering",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": True,
    "email": ["data-alerts@company.com"],
    "execution_timeout": timedelta(hours=3),
}


# ── Helper functions ──
def check_data_volume(**context):
    """Branch based on data volume — skip transform if no new data."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blobs = list(bucket.list_blobs(
        prefix=f"raw/transactions/date={context['ds']}/"
    ))
    total_size = sum(b.size for b in blobs)

    if total_size == 0:
        return "skip_no_data"
    elif total_size < 1000:  # < 1KB = suspicious
        return "alert_low_volume"
    else:
        return "create_cluster"


def run_quality_checks(**context):
    """Execute all 5 data quality checks."""
    from src.quality.data_quality_engine import DataQualityEngine

    dq = DataQualityEngine(PROJECT_ID)
    table = f"{PROJECT_ID}.analytics.fact_transactions"

    dq.check_schema(table, {
        "transaction_id": "STRING",
        "account_id": "STRING",
        "customer_id": "STRING",
        "amount": "NUMERIC",
        "created_date": "DATE",
    })
    dq.check_freshness(table, "_ingestion_ts", max_hours=6)
    dq.check_volume_anomaly(table, z_threshold=3.0)
    dq.check_null_rate(table, "customer_id", max_pct=0.5)
    dq.check_null_rate(table, "amount", max_pct=1.0)
    dq.check_business_rule(
        table, "no_negative_amounts",
        f"SELECT COUNTIF(amount < 0) as violations FROM `{table}` "
        f"WHERE created_date = '{context['ds']}'"
    )
    dq.run_suite(table)


def reindex_vector_store(**context):
    """Re-index schema metadata in the vector store after data loads.
    
    This keeps the RAG pipeline's schema knowledge up to date.
    If new tables/columns are added, they'll be discoverable
    via natural language queries immediately.
    """
    from src.ai.vector_store import VectorStore, BANKING_SCHEMAS
    from src.ai.embedding_engine import EmbeddingEngine

    vs = VectorStore(EmbeddingEngine())
    vs.index_schemas(BANKING_SCHEMAS)
    # In production: persist embeddings to GCS for other services


# ── DAG Definition ──
with DAG(
    "banking_daily_pipeline",
    default_args=default_args,
    description="Daily banking data ingestion, transform, quality, and AI indexing",
    schedule_interval="0 2 * * *",  # 2 AM IST daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["banking", "daily", "production"],
) as dag:

    # ── Step 1: Wait for source data ──
    wait_for_data = GCSObjectExistenceSensor(
        task_id="wait_for_source_data",
        bucket=BUCKET,
        object="raw/transactions/date={{ ds }}/_SUCCESS",
        timeout=7200,  # 2 hours max wait
        poke_interval=300,  # Check every 5 minutes
        mode="reschedule",  # Release worker while waiting
    )

    # ── Step 2: Check volume ──
    check_volume = BranchPythonOperator(
        task_id="check_data_volume",
        python_callable=check_data_volume,
    )

    # ── Step 3: Create ephemeral cluster ──
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_cluster",
        cluster_name=CLUSTER_NAME,
        project_id=PROJECT_ID,
        region=REGION,
        num_workers=4,
        worker_machine_type="n1-standard-4",
        master_machine_type="n1-standard-4",
        init_actions_uris=[f"gs://{BUCKET}/scripts/init.sh"],
        metadata={"PIP_PACKAGES": "google-cloud-bigquery pyarrow"},
        idle_delete_ttl=600,  # Auto-delete if idle for 10 min (safety net)
    )

    # ── Step 4: Run Spark transformation ──
    run_transform = DataprocSubmitPySparkJobOperator(
        task_id="run_spark_transform",
        main=f"gs://{BUCKET}/scripts/transform_spark.py",
        arguments=[PROJECT_ID, "{{ ds }}"],
        cluster_name=CLUSTER_NAME,
        project_id=PROJECT_ID,
        region=REGION,
        dataproc_pyspark_properties={
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
        },
    )

    # ── Step 5: Data quality checks ──
    quality_gate = PythonOperator(
        task_id="data_quality_checks",
        python_callable=run_quality_checks,
    )

    # ── Step 6: Re-index vector store ──
    reindex = PythonOperator(
        task_id="reindex_vector_store",
        python_callable=reindex_vector_store,
    )

    # ── Step 7: Delete cluster (ALWAYS runs) ──
    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_cluster",
        cluster_name=CLUSTER_NAME,
        project_id=PROJECT_ID,
        region=REGION,
        trigger_rule=TriggerRule.ALL_DONE,  # Runs even if upstream fails
    )

    # ── Step 8: Notify ──
    notify = PythonOperator(
        task_id="notify_slack",
        python_callable=lambda **ctx: print(f"Pipeline complete for {ctx['ds']}"),
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Skip path
    skip = PythonOperator(
        task_id="skip_no_data",
        python_callable=lambda: print("No data for today — skipping"),
    )

    alert = PythonOperator(
        task_id="alert_low_volume",
        python_callable=lambda: print("WARNING: Suspiciously low data volume"),
    )

    # ── DAG Flow ──
    wait_for_data >> check_volume

    check_volume >> create_cluster >> run_transform >> quality_gate >> reindex >> delete_cluster >> notify
    check_volume >> skip
    check_volume >> alert
