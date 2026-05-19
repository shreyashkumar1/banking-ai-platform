"""
PySpark Transformation Pipeline — Heavy data processing on Dataproc.

Handles: deduplication, multi-table joins, feature engineering, aggregations.
Optimized with AQE, broadcast joins, and salted keys for data skew.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, DateType
import logging

logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "banking-ai-transform") -> SparkSession:
    """Create optimized Spark session with AQE and performance tuning."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "500") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.sql.broadcastTimeout", "600") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()


def deduplicate(df: DataFrame, id_col: str, ts_col: str) -> DataFrame:
    """Remove duplicate records, keeping the most recent by timestamp.
    
    Why: Source systems (APIs, event buses) often send duplicate events.
    At-least-once delivery guarantees in Pub/Sub mean duplicates are expected.
    """
    window = Window.partitionBy(id_col).orderBy(F.desc(ts_col))
    return df \
        .withColumn("_row_num", F.row_number().over(window)) \
        .filter(F.col("_row_num") == 1) \
        .drop("_row_num")


def broadcast_join(large_df: DataFrame, small_df: DataFrame,
                   join_key: str, how: str = "left") -> DataFrame:
    """Broadcast join — send small table to all executors, no shuffle.
    
    Use when: small_df < 5M rows (fits in executor memory).
    Benefit: Eliminates expensive network shuffle.
    """
    return large_df.join(F.broadcast(small_df), on=join_key, how=how)


def salted_join(left_df: DataFrame, right_df: DataFrame,
                join_key: str, salt_range: int = 10) -> DataFrame:
    """Salted join for extreme data skew.
    
    When one join key has 100x more records than others,
    add a random salt to distribute the hot key across partitions.
    
    Cost: right_df is replicated salt_range times.
    Benefit: Hot key processing parallelized across salt_range executors.
    """
    spark = left_df.sparkSession

    # Add random salt to left (large) table
    salted_left = left_df \
        .withColumn("_salt", (F.rand() * salt_range).cast("int")) \
        .withColumn("_salted_key", F.concat(F.col(join_key), F.lit("_"), F.col("_salt")))

    # Explode right (small) table with all salt values
    salt_df = spark.range(salt_range).withColumnRenamed("id", "_salt")
    exploded_right = right_df.crossJoin(salt_df) \
        .withColumn("_salted_key", F.concat(F.col(join_key), F.lit("_"), F.col("_salt")))

    # Join on salted key
    result = salted_left.join(exploded_right, on="_salted_key", how="left")

    # Clean up salt columns
    return result.drop("_salt", "_salted_key")


def compute_transaction_features(transactions: DataFrame) -> DataFrame:
    """Engineer features for downstream analytics and fraud detection.
    
    Window functions compute per-customer rolling metrics:
    - Running total: Cumulative transaction amount
    - Transaction rank: Most recent first
    - Daily velocity: Number of transactions per day
    - Amount percentile: Where this transaction falls in customer's history
    """
    customer_window = Window.partitionBy("customer_id").orderBy("created_timestamp")
    daily_window = Window.partitionBy("customer_id", "created_date")

    return transactions \
        .withColumn("running_total", F.sum("amount").over(customer_window)) \
        .withColumn("txn_rank", F.row_number().over(
            Window.partitionBy("customer_id").orderBy(F.desc("created_timestamp"))
        )) \
        .withColumn("daily_txn_count", F.count("*").over(daily_window)) \
        .withColumn("daily_total_amount", F.sum("amount").over(daily_window))


def run_daily_transform(project_id: str, date: str):
    """Main daily transformation pipeline.
    
    Reads raw data from GCS, deduplicates, joins with dimensions,
    engineers features, and writes to BigQuery.
    """
    spark = create_spark_session()

    logger.info(f"Starting daily transform for {date}")

    # ── Read raw data ──
    transactions = spark.read.parquet(f"gs://{project_id}-raw/transactions/date={date}/")
    customers = spark.read.parquet(f"gs://{project_id}-raw/customers/")
    accounts = spark.read.parquet(f"gs://{project_id}-raw/accounts/")

    logger.info(f"Raw transactions: {transactions.count()} rows")

    # ── Deduplicate ──
    deduped = deduplicate(transactions, "transaction_id", "created_timestamp")
    logger.info(f"After dedup: {deduped.count()} rows")

    # ── Join with dimensions ──
    # Customers: ~500K rows → broadcast join (no shuffle)
    enriched = broadcast_join(deduped, customers, "customer_id")

    # Accounts: ~1M rows → broadcast join
    enriched = broadcast_join(enriched, accounts, "account_id")

    # ── Feature engineering ──
    featured = compute_transaction_features(enriched)

    # ── Write to BigQuery ──
    featured \
        .repartition(100, "created_date") \
        .write.format("bigquery") \
        .option("table", f"{project_id}.analytics.fact_transactions") \
        .option("partitionField", "created_date") \
        .option("clusteredFields", "customer_id,transaction_type") \
        .mode("append") \
        .save()

    logger.info(f"Daily transform complete: {featured.count()} rows written")

    spark.stop()


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "banking-ai-prod"
    date = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
    run_daily_transform(project, date)
