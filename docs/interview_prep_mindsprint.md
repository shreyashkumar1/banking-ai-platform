# Interview Prep — Mindsprint: MedAdvisor Healthcare Data Platform

## Mindsprint | Data & AI Engineer | Oct 2024 – Present
### Client: MedAdvisor (Healthcare/Pharma Analytics)

---

## 1. Elevator Pitch (30 seconds)

> "At Mindsprint I'm building MedAdvisor's healthcare data platform — they provide medication adherence analytics to pharmaceutical companies, pharmacy chains, and health insurers across Australia and the UK. The biggest thing I solved: their core metric, PDC — proportion of days covered — was being calculated at 78% accuracy because it couldn't handle overlapping prescription fills, drug switching between brands and generics, or patients filling at multiple pharmacies. I rebuilt it with a timeline-based approach that adjusts for overlaps and groups by molecule level, bringing accuracy to 96%. I also built the HL7/FHIR ingestion layer for 5,000+ pharmacies and a drug interaction detection system that caught 34 critical patient safety alerts in the first month."

---

## 2. The Problems Worth Talking About

### Problem 1 — PDC Calculation Was Wrong (The Core Business Metric)

**Situation:** PDC (Proportion of Days Covered) is the #1 metric in medication adherence. It measures what percentage of days a patient actually had their medication available. A patient is "adherent" at PDC ≥ 80%. Below that, hospitalisation risk jumps — costing insurers $10K–$50K per admission. Every percentage point of PDC improvement across a drug class saves millions.

The existing calculation was naive — it summed days_supply across all fills and divided by the observation period. This broke in three common cases:

1. **Overlapping fills:** Patient fills a 30-day prescription on Jan 1, then fills again on Jan 28 (3-day overlap because they have leftover pills). The old method counted 60 days covered for 58 actual days.
2. **Drug switching:** Patient switches from brand Metformin to generic. Different NDC codes but same molecule. Old system treated these as different drugs — showed the patient as non-adherent on BOTH.
3. **Multi-pharmacy fills:** Same drug filled at two pharmacies in the same week. Old system double-counted.

**The fix:** Timeline-based PDC. Build a day-by-day covered/uncovered timeline per patient per molecule. Adjust overlapping fills by pushing the second fill's effective start to after the first fill ends. Group by ATC-5 code (molecule level) instead of NDC (product level) so brand-to-generic switches are counted as continuous therapy.

**Result:** PDC accuracy: 78% → 96% (validated against manual chart review by the clinical team). At-risk patients flagged 30+ days earlier because we now use a rolling 90-day window instead of calculating only at period end.

### Problem 2 — HL7/FHIR: Healthcare's Ugly Data Formats

**Situation:** 5,000+ pharmacies send data. Legacy systems send HL7v2 pipe-delimited messages (a format designed in 1987 that looks like: `MSH|^~\&|PHARMSYS|PHARMACY_001||...`). Modern EMRs send FHIR R4 JSON. Sometimes the same pharmacy sends both formats on different days.

**Why this is harder than normal data ingestion:** Healthcare messages embed clinical semantics. An HL7 `RXD` segment contains the drug dispensed, but the drug code might be in the 2nd or 3rd component of a composite field, separated by `^`. A FHIR R4 MedicationDispense resource nests the drug code inside `medicationCodeableConcept.coding[0].code`. Both must map to the same `{patient_hash, ndc_code, fill_date, days_supply}` output schema.

**The fix:** A Dataflow DoFn that detects format by inspecting the first characters (`MSH|` = HL7, `{` = FHIR), routes to the appropriate parser, and outputs a unified schema. Invalid NDC codes and parse errors go to dead-letter for ops review.

### Problem 3 — Drug Interaction Detection Across Pharmacies

**Situation:** A patient takes Warfarin (blood thinner) from Pharmacy A and gets prescribed Aspirin from Pharmacy B. Each pharmacist checks their own system — no interaction flagged. But together, it's a dangerous bleeding risk.

**The fix:** Self-join on the active prescriptions table — patient has drug A AND drug B simultaneously, where (A, B) is in the WHO-UMC drug interaction database. Broadcast the interaction reference table (it's small — ~50K known pairs). Critical interactions trigger alerts to pharmacies and prescribers.

**Result:** 1,200+ interaction alerts in month 1. 34 were CRITICAL severity.

---

## 3. Technical Deep-Dives

**Q: Walk through the PDC timeline calculation step by step.**

A: 
1. Pull all prescriptions for a patient for drug X (grouped by ATC-5, not NDC, so brand and generic count as the same drug).
2. Sort by fill_date ascending.
3. For each fill, calculate the "adjusted start" — if this fill's date is before the previous fill's end date (overlap), push the start to the day after the previous fill ends. This prevents double-counting.
4. Calculate adjusted_days_covered = adjusted_end - adjusted_start for each fill.
5. Sum adjusted_days_covered. Divide by evaluation period (typically 90 days). Cap at 1.0.
6. If PDC < 0.80 → non-adherent. Between 0.60-0.80 → at-risk. ≥ 0.80 → adherent.

The key insight: window function `LAG(date_add(fill_date, days_supply)) OVER (PARTITION BY patient, atc_code ORDER BY fill_date)` gives us the previous fill's end date. Then `GREATEST(fill_date, prev_fill_end)` is the adjusted start. Handles any number of overlapping fills correctly.

**Q: Why ATC-5 grouping instead of NDC for PDC?**

A: NDC (National Drug Code) identifies a specific product — Metformin 500mg by manufacturer X is a different NDC than Metformin 500mg by manufacturer Y. But they're the same molecule with the same clinical effect. ATC-5 (WHO Anatomical Therapeutic Chemical classification, 5th level) groups by chemical substance. All Metformin products share ATC code A10BA02 regardless of manufacturer. When a patient switches from brand to generic (extremely common — insurer formulary changes force this), ATC-5 correctly counts it as continuous therapy. NDC would show a "gap" and flag the patient as non-adherent.

**Q: How do you handle the HL7v2 composite field parsing?**

A: HL7v2 uses multi-level delimiters defined in the MSH segment: `|` separates fields, `^` separates components within a field, `~` separates repetitions, `\` is escape. For the RXD (dispense) segment, the drug code is typically in field 2, which is a composite: `NDC_12345^METFORMIN 500MG^NDC`. We split on `^` — first component is the code, second is the display name, third is the coding system. But some pharmacy systems put it in a different order, or use different segment types. We handle this with a configurable parser that takes a mapping of (segment_type, field_index, component_index) per source system.

**Q: How does the drug interaction self-join work without exploding?**

A: The key is the join condition `a.atc_code_4 < b.atc_code_4`. This prevents duplicate pairs: (Warfarin, Aspirin) and (Aspirin, Warfarin) are the same interaction — the `<` condition keeps only one. Without it, you'd get N² pairs. The interaction reference table is broadcast (small — ~50K rows). The self-join on active_prescriptions is partitioned by patient_hash, so each partition handles one patient's drugs independently.

**Q: Why column-level masking instead of just separate tables for PII vs analytics?**

A: Separate tables create a sync problem — analytics table can drift from PII table. Column-level masking keeps one table with dynamic visibility: the same query returns `customer_name = 'MASKED'` for general analysts and the real name for data stewards. This is enforced at BigQuery level using `SESSION_USER()`, so it can't be bypassed by the application layer. For PHI (Protected Health Information) specifically, we use irreversible hashing (SHA-256 with salt) for patient_id — even data stewards can't reverse it to a real Medicare number.

**Q: What's the adverse event SLA monitoring approach?**

A: The TGA (Australia's FDA equivalent) requires adverse drug reaction reports within 15 calendar days. Our system calculates `days_remaining = 15 - datediff(current_date, event_date)` for every unsubmitted report. Priority: RED (≤2 days), AMBER (3-5 days), GREEN (>5 days). RED triggers Slack + PagerDuty alerts to the pharmacovigilance team. The Airflow DAG runs this check daily at 8 AM AEST. One near-miss (4 hours before deadline) in the 6 months before we built this — zero since.

---

## 4. Architecture Decisions

**Q: Why Dataflow for ingestion instead of just loading files to GCS?**

A: Two reasons specific to healthcare. First, HL7 messages arrive as events (not files) — they need stateless streaming processing, which Beam handles natively. Second, dead-letter routing is critical for healthcare — you can't silently drop a malformed prescription record. Beam's `TaggedOutput` sends parse failures to a separate path where the ops team reviews them. A dropped record could mean a missed drug interaction.

**Q: Why BigQuery for healthcare instead of a healthcare-specific OLAP?**

A: MedAdvisor's analytics are SQL-based — PDC calculations, cohort analysis, drug class comparisons. BigQuery handles these at scale without managing infrastructure. The alternative (Snowflake) would work too, but the client was already on GCP with Pub/Sub and Dataflow. Keeping the entire stack on one cloud reduces data transfer costs and operational complexity. Healthcare data volumes (5K pharmacies × ~100 prescriptions/pharmacy/day = 500K events/day) are well within BigQuery's sweet spot.

---

## 5. Behavioral Questions

**Q: Tell me about a time you solved a problem that the business didn't know was a problem.**

A: The drug interaction detection. MedAdvisor's business was about adherence — they weren't in the patient safety alerting space. But when I was building the prescriptions pipeline, I noticed patients filling contraindicated drugs at different pharmacies. I built a POC: self-join on active prescriptions against the WHO interaction database. Found 34 critical interactions in month one. The clinical team was shocked — these were real patients at real risk. MedAdvisor is now exploring this as a new product feature for pharmacy chains.

**Q: How do you handle working with healthcare domain knowledge you didn't have?**

A: I didn't know what PDC was when I started. The clinical team explained the concept in 30 minutes. Then I spent 2 days reading the PQA (Pharmacy Quality Alliance) specification for PDC calculation — it's a 15-page document that defines exactly how to handle overlaps, switching, and gaps. The technical implementation was straightforward once I understood the clinical logic. The lesson: don't try to learn an entire domain — learn the specific metrics and rules that your pipeline needs to produce correctly.

---

## 6. Numbers to Remember

| Metric | Before | After |
|---|---|---|
| PDC accuracy | 78% | 96% |
| At-risk patient flagging | End of 90-day period | 30+ days earlier (rolling) |
| Drug interactions detected | Manual pharmacist review | 1,200+ automated/month (34 critical) |
| Formulary query response | 2-3 hours | Under 2 minutes |
| Adverse event SLA misses | 1 near-miss in 6 months | Zero |
| Data format support | CSV only | HL7v2 + FHIR R4 + CSV |
| Bad-record rate | 3-5% | <1% |
| Pipeline runtime | ~2 hours | ~45 minutes |
