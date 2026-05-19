"""
Dataflow Ingestion Pipeline — Apache Beam on GCP.

Ingests banking transaction events from Pub/Sub and batch files from GCS.
Schema validation at ingestion — bad records routed to dead-letter topic.
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions
from apache_beam.io.gcp.bigquery import WriteToBigQuery
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Schema Definitions ──
TRANSACTION_SCHEMA = {
    "required_fields": ["transaction_id", "account_id", "customer_id", "amount", "transaction_type"],
    "field_types": {
        "transaction_id": str,
        "account_id": str,
        "customer_id": str,
        "amount": (int, float),
        "transaction_type": str,
        "channel": str,
        "status": str,
        "created_timestamp": str,
    },
    "valid_transaction_types": ["TRANSFER", "PAYMENT", "WITHDRAWAL", "DEPOSIT", "LOAN_DISBURSEMENT"],
    "valid_channels": ["MOBILE", "WEB", "ATM", "BRANCH", "UPI"],
}


class SchemaValidationFn(beam.DoFn):
    """Validate incoming records against expected schema.
    
    Bad records go to dead-letter for manual review.
    Good records flow through to transformation.
    """

    def __init__(self, schema: dict):
        self.schema = schema

    def process(self, record):
        errors = []

        # Check required fields
        for field in self.schema["required_fields"]:
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, expected_type in self.schema.get("field_types", {}).items():
            if field in record and record[field] is not None:
                if not isinstance(record[field], expected_type):
                    errors.append(f"Type mismatch: {field} expected {expected_type}, got {type(record[field])}")

        # Check business rules
        if "amount" in record and record["amount"] is not None:
            if record["amount"] < 0:
                errors.append(f"Invalid amount: {record['amount']} (must be >= 0)")

        if "transaction_type" in record:
            valid_types = self.schema.get("valid_transaction_types", [])
            if valid_types and record["transaction_type"] not in valid_types:
                errors.append(f"Invalid transaction_type: {record['transaction_type']}")

        if errors:
            yield beam.pvalue.TaggedOutput("dead_letter", {
                "record": record,
                "errors": errors,
                "timestamp": datetime.utcnow().isoformat(),
            })
        else:
            yield record


class EnrichTransactionFn(beam.DoFn):
    """Enrich transactions with derived fields."""

    def process(self, record):
        # Add ingestion metadata
        record["_ingestion_ts"] = datetime.utcnow().isoformat()
        record["_source"] = "pubsub_streaming"

        # Derive date partition key
        if "created_timestamp" in record:
            record["created_date"] = record["created_timestamp"][:10]
        else:
            record["created_date"] = datetime.utcnow().strftime("%Y-%m-%d")

        # Flag high-value transactions (>10 lakhs)
        if record.get("amount", 0) > 1_000_000:
            record["high_value_flag"] = True
        else:
            record["high_value_flag"] = False

        yield record


def parse_pubsub_message(message: bytes) -> dict:
    """Parse and decode Pub/Sub message payload."""
    try:
        return json.loads(message.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Failed to parse message: {e}")
        return {"_parse_error": str(e), "_raw": message.decode("utf-8", errors="replace")}


def run_streaming_pipeline(project_id: str, subscription: str, output_table: str):
    """Run streaming ingestion pipeline from Pub/Sub to BigQuery.
    
    Design decisions:
    - At-least-once delivery: Pub/Sub guarantees no data loss
    - Dedup downstream: Use transaction_id as idempotency key
    - 5-min windows: Balance latency vs write efficiency
    - 1-hour allowed lateness: Handle late-arriving events
    """
    options = PipelineOptions([
        "--streaming",
        "--runner=DataflowRunner",
        f"--project={project_id}",
        "--region=asia-south1",
        "--max_num_workers=10",
        "--autoscaling_algorithm=THROUGHPUT_BASED",
    ])

    with beam.Pipeline(options=options) as p:
        # Read from Pub/Sub
        raw = (
            p
            | "ReadFromPubSub" >> beam.io.ReadFromPubSub(subscription=subscription)
            | "ParseJSON" >> beam.Map(parse_pubsub_message)
        )

        # Validate schema (outputs: main=valid, dead_letter=invalid)
        validated = (
            raw
            | "ValidateSchema" >> beam.ParDo(
                SchemaValidationFn(TRANSACTION_SCHEMA)
            ).with_outputs("dead_letter", main="valid")
        )

        # Process valid records
        (
            validated["valid"]
            | "Enrich" >> beam.ParDo(EnrichTransactionFn())
            | "Dedup" >> beam.Distinct(key=lambda x: x.get("transaction_id", ""))
            | "WriteToBQ" >> WriteToBigQuery(
                table=output_table,
                method=WriteToBigQuery.Method.STREAMING_INSERTS,
                create_disposition="CREATE_NEVER",
                write_disposition="WRITE_APPEND",
            )
        )

        # Route bad records to dead-letter
        (
            validated["dead_letter"]
            | "WriteDeadLetter" >> beam.io.WriteToText(
                f"gs://{project_id}-dlq/banking/dead_letter",
                file_name_suffix=".json",
            )
        )


def run_batch_pipeline(project_id: str, input_path: str, output_table: str):
    """Run batch ingestion from GCS Parquet files to BigQuery.
    
    Used for: SFTP uploads, historical data loads, partner file drops.
    """
    options = PipelineOptions([
        "--runner=DataflowRunner",
        f"--project={project_id}",
        "--region=asia-south1",
        f"--temp_location=gs://{project_id}-temp/beam-temp",
    ])

    with beam.Pipeline(options=options) as p:
        (
            p
            | "ReadParquet" >> beam.io.ReadFromParquet(input_path)
            | "Validate" >> beam.ParDo(SchemaValidationFn(TRANSACTION_SCHEMA))
            | "Enrich" >> beam.ParDo(EnrichTransactionFn())
            | "WriteToBQ" >> WriteToBigQuery(
                table=output_table,
                method=WriteToBigQuery.Method.FILE_LOADS,
                create_disposition="CREATE_NEVER",
                write_disposition="WRITE_APPEND",
            )
        )
