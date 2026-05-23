# Shreyash Kumar
Bengaluru, Karnataka, India | +91 9131613637 | shreyashkumar456@gmail.com | [GitHub Portfolio](https://github.com/shreyashkumar1)

---

## PROFESSIONAL SUMMARY
Senior Data & AI Engineer with 4+ years of experience designing, building, and optimizing high-throughput data platforms on Google Cloud Platform (GCP). Expert in Apache Spark (PySpark), Apache Beam (Dataflow), BigQuery, and building secure, production-grade RAG and Agentic AI applications. Proven track record in clinical data engineering (HL7/FHIR, ATC classifications, PDC metrics) at Mindsprint and high-volume grocery retail analytics (Sainsbury's UK card loyalty platforms, store replenishment systems) at TCS.

---

## TECHNICAL SKILLS
* **Languages & Core**: Python, SQL, Bash
* **Data Engineering**: Apache Spark, Apache Beam (Dataflow), Dataproc, Cloud Composer (Airflow), BigQuery, GCS, Pub/Sub, Kafka
* **Spark Optimization**: AQE tuning, Broadcast joins, Key Salting for Data Skew, Partitioning & Clustering
* **AI & LLMs**: RAG Pipelines, Autonomous Agents (ReAct framework), Vector Databases, Semantic Schema Search, LLM Security (SQL injection prevention)
* **Healthcare Standards**: HL7v2 parsing, FHIR R4 resources, ATC/NDC drug code ontologies, PHI De-identification & HIPAA/GDPR compliance
* **DevOps & Infrastructure**: Docker, GitHub Actions CI/CD, Terraform, Logging & Alerting (Slack, PagerDuty integration)

---

## EXPERIENCE

### Data & AI Engineer | Mindsprint (Bengaluru, India)
**Oct 2024 – Present** | *Client: MedAdvisor (Healthcare & Pharmacy Analytics)*
* Developed a timeline-based Proportion of Days Covered (PDC) pipeline on Dataproc PySpark to calculate medication adherence; mapped drugs to ATC-5 level molecule categories to resolve overlaps and generic-brand switching, boosting calculation accuracy from 78% to 96%.
* Designed a unified Dataflow parsing pipeline that ingests legacy HL7v2 pipe-delimited messages and modern FHIR R4 JSON resources from 5,000+ pharmacies, standardizing them into a single schema with dead-letter queue (DLQ) routing for malformed data.
* Built a prescription self-join engine in PySpark that matches active patient medications against the WHO-UMC contraindication database, triggering ~1,200 automated drug-drug interaction alerts monthly.
* Implemented a natural language query interface (Gemini 1.5 Pro + Vector Store schema indexing) that translates English questions to SQL queries over insurer formularies, reducing clinical account manager lookup times from 2 hours to under 2 minutes.
* Automated SLA compliance tracking for mandatory TGA/MHRA adverse drug event reporting within a 15-day window, eliminating compliance near-misses.
* Enforced rigid testing and quality standards by setting up CI/CD workflows using Pytest, Ruff, and Mypy, maintaining >90% test coverage and automatic PHI column masking in BigQuery.

### Data Engineer | Tata Consultancy Services (Bengaluru, India)
**Aug 2021 – Oct 2024** | *Client: Sainsbury's (UK Grocery Retailer)*
* Architected a real-time EPOS transaction ingestion pipeline using Pub/Sub and Dataflow (Apache Beam) to handle Till scan events from 1,400+ stores; reduced data replication lag from 6 hours to under 5 minutes, cutting fresh produce wastage by 18% in pilot stores.
* Optimized PySpark loyalty joins for 18 million active Nectar cardholders by pre-aggregating transaction events to customer-day levels and applying key salting, resolving 200:1 join skew issues and cutting job execution times from 5.2 hours to 48 minutes.
* Engineered an automated month-end supplier settlement reconciliation pipeline in PySpark that pro-rates promotional allowances against actual depot goods received (GRN) sheets, reducing monthly close cycles from 10 days to 2 days and recovering £340K in short-shipments in Q1.
* Structured role-based and regional row-level access security (RLS) policies along with column-level masking in BigQuery to prevent unauthorized access to customer PII and maintain GDPR compliance.
* Reduced cloud infrastructure compute bills by £6.8K/month by replacing always-on Dataproc clusters with ephemeral, autoscaling Dataproc cluster pools utilizing preemptible VMs, configured with Airflow teardown tasks to run even on pipeline exceptions.

---

## SELECTED PROJECTS

### Banking AI Platform — SBI-Scale Intelligent Data & AI Infrastructure
*Developed a production-grade, secure, and compliant data & AI platform designed to handle SBI-scale transaction volume.*
* **Autonomous Investigation Agent**: Designed a ReAct reasoning agent with state loop detection, parameter verification, and compliance-ready audit trails to analyze UPI velocity fraud patterns and account takeovers.
* **NL-to-SQL RAG Engine**: Implemented a natural language query interface with a semantic vector search index over BigQuery schemas, including a static analysis SQL validator that intercepts and blocks DML/DDL injections (e.g. DROP/DELETE) before database execution.
* **Provider-Agnostic AI Integration**: Designed a configuration-driven LLM layer supporting OpenAI, Gemini, and self-hosted models, complying with local Reserve Bank of India (RBI) data residency regulations.
* **Data Quality Gateways**: Built a 5-step automated validation engine (validating schemas, freshness SLAs, null rate thresholds, and daily volume anomalies using 30-day Z-scores) blocking bad data loads dynamically.

---

## EDUCATION
* **B.Tech in Computer Science & Engineering** | SRM Institute of Science and Technology, Chennai (2021)

---

## CERTIFICATIONS
* **Google Cloud Professional Data Engineer**
* **Google Cloud Professional Cloud Architect**
