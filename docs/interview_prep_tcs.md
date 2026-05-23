# Interview Prep — TCS: Sainsbury's Retail Data Platform

## Client: Sainsbury's (UK's 2nd Largest Grocer) | Aug 2021 – Oct 2024

---

## 1. Elevator Pitch (30 seconds)

> "At TCS I built the data platform for Sainsbury's — 1,400 stores, 24 million transactions a week, 18 million Nectar loyalty members. Three big wins: I fixed a store replenishment lag that was causing 25% waste on fresh produce by streaming EPOS data in near-real-time; I resolved a 200:1 data skew on Nectar loyalty joins by pre-aggregating before joining — cut the pipeline from 5 hours to 48 minutes; and I automated supplier settlement reconciliation across 3,000 suppliers from 10 days to 2 days, recovering £340K in missed deductions in the first quarter."

---

## 2. The Problems Worth Talking About

### Problem 1 — Fresh Produce Waste (The Perishability Problem)

**Situation:** Grocery is unique because of shelf life. A laptop sits on a shelf for months. A ready meal has 3 days. Bakery croissants have 8 hours. Store replenishment ran on 6-8 hour stale data. By the time the system knew an item had sold out, the morning rush was over. Overstocking on short-shelf items: 25% waste. Stockouts on promoted items: 12% lost revenue.

**Why batch couldn't be fixed:** The overnight Spark job reads the entire EPOS extract. It takes 40 minutes. Running it every 30 minutes means it never finishes before the next run starts.

**The fix:** Streaming. EPOS tills already publish events to a message bus. Connected that to Pub/Sub → Dataflow. Store managers now see sales velocity updated every 5 minutes. The tricky part was handling grocery-specific data — weighed items use store-generated barcodes (format: 2PPPPPPWWWWWC) that need to be parsed differently from standard EAN-13.

**Result:** Data lag: 6-8 hours → ~5 minutes. Fresh produce waste in pilot stores: 25% → 18%.

### Problem 2 — Nectar Loyalty Skew (200:1 Power Shoppers)

**Situation:** 18M Nectar cardholders, but the top 500 "power shoppers" have 1,500+ transactions/month. When you join sales to Nectar events on nectar_card_id, those 500 users create a 200:1 skew. Standard Spark OOMs.

**What I tried first:**
- AQE skew handling: threshold is 5x median. At 200x, even split partitions are 40x larger than normal — OOM continues.
- 16GB executor memory: delayed OOM by 20 minutes. Same outcome.

**What worked:** Pre-aggregate Nectar events to customer-day level BEFORE the join. A power shopper with 1,500 transactions becomes ONE row per day. Join ratio goes from 1:200 to 1:1. Broadcast the aggregated Nectar table (~800K rows for a day).

### Problem 3 — Supplier Settlement (10 Days to Close Month-End)

**Situation:** Sainsbury's has 3,000+ suppliers. Monthly settlement requires matching purchase orders → delivery confirmations → promotional allowances → markdown claims. Four different systems. Finance team spent 10 days on spreadsheets.

**The hard part:** Short-shipments. Sainsbury's orders 1000 Heinz Beans, Heinz ships 950. Sainsbury's pays for 950, but the promotional allowance was calculated on 1000. The allowance must be pro-rated to 950/1000 = 95%. This pro-rating was being done manually — and finance missed it on ~15% of short-shipped orders.

**Result:** 10 days → 2 days. £340K recovered in Q1 from short-shipment deductions that were being missed.

---

## 3. Technical Deep-Dives

**Q: Why Pub/Sub → Dataflow for EPOS instead of Kafka → Spark Streaming?**

A: Sainsbury's was already on GCP. Pub/Sub is managed — zero ops overhead. Kafka requires ZooKeeper, broker sizing, partition management, replication factor tuning. For a grocery retailer that wants to focus on selling food, not managing infrastructure, serverless wins. Dataflow also scales to zero when stores close at night — no idle cost.

**Q: How did you handle the weighed-item barcodes?**

A: Standard products use EAN-13 barcodes (global, unique per SKU). But weighed items (fruit, deli meat, cheese) get store-generated barcodes: `2PPPPPPWWWWWC` where P is an internal product code and W is weight in grams. I detect these by prefix "2", extract the product code and weight from the barcode itself, and join to a separate weighed-product dimension table instead of the standard SKU master.

**Q: Explain the pre-aggregation approach for the Nectar skew in detail.**

A: Instead of joining 80M raw Nectar events to 160M EPOS transactions (skewed on nectar_card_id), I aggregate both sides to customer-day level first. Nectar: `GROUP BY nectar_card_id, event_date → points_earned, points_redeemed, partner_count`. Sales: `GROUP BY nectar_card_id, sale_date → daily_spend, items_purchased, stores_visited, categories`. Now both sides have ~800K rows for a typical day — broadcastable. The 1,500-transaction power shopper is one row. Zero skew.

**Q: What's `trigger_rule=ALL_DONE` on the delete cluster task?**

A: Airflow default is `ALL_SUCCESS` — a task only runs if all upstream tasks succeeded. If Spark fails, the cluster delete task would be SKIPPED, leaving a live Dataproc cluster burning money 24/7. `ALL_DONE` means "run this task regardless of upstream outcome." It's the one task where you explicitly don't care about success/failure upstream — the cluster MUST be cleaned up.

**Q: How did you handle GDPR for Nectar customer data?**

A: Two layers. Column-level masking: customer_name and postcode are masked for general analysts (they see 'MASKED' and 'SW1 ***'). Only the data_stewards security group sees full values. Row-level security: a London analyst querying fact_sales automatically gets filtered to `WHERE store_region = 'london'`. Both enforced at BigQuery level using row access policies and authorized views.

---

## 4. Behavioral Questions

**Q: Tell me about a production incident during a critical period.**

A: Sainsbury's had a major promotion launch on a Saturday morning. The Nectar loyalty pipeline from the night before had failed at 3 AM — the executor handling the power shoppers OOMed. By 7 AM the marketing team couldn't see Nectar point allocations for the promotion. I checked executor logs, identified the skewed partition, added a quick pre-filter on Nectar events to only include today's active cards, and re-ran. Finished by 9:30 AM. That weekend incident is what led me to design the proper pre-aggregation fix over the following two weeks.

**Q: How did you convince stakeholders to invest in streaming?**

A: The 25% waste number on fresh produce did it. I calculated the annual cost: 25% waste × fresh produce revenue × margin impact = roughly £4-5M per year in preventable loss. Streaming pipeline development: 6 weeks, no additional infrastructure cost (Dataflow is serverless). The ROI was obvious. They approved it the same week.

---

## 5. Numbers to Remember

| Metric | Before | After |
|---|---|---|
| Store replenishment data lag | 6-8 hours | ~5 minutes |
| Fresh produce waste (pilot) | 25% | 18% |
| Nectar loyalty pipeline | 5.2 hours | 48 minutes |
| Supplier settlement | 10 days | 2 days |
| Short-shipment savings (Q1) | Untracked | £340K recovered |
| Cluster cost | £8K/month | £1.2K/month |
| GDPR compliance | PII accessible to all | Column + row level security |
