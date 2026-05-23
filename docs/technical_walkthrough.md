# Technical Code Walkthrough — Banking AI Platform

This document provides a line-by-line explanation of the core files in the `banking-ai-platform` AI stack. It explains the software engineering design choices, safety constraints, and implementation details of the natural language query pipeline and the autonomous investigation agent.

---

## 1. RAG Query Engine (`src/ai/rag_query_engine.py`)

The query engine handles natural language to SQL translation (NL-to-SQL) for database queries. In a banking context, security and reliability are paramount. The pipeline consists of:
1. **Semantic Schema Retrieval**: Finding the tables and columns relevant to the user query using vector search.
2. **SQL Generation and Validation**: Generating standard BigQuery SQL using an LLM and running a strict static analysis pass to prevent injection attacks and DML/DDL modifications.
3. **Self-Healing Retry**: Feeding BigQuery syntax or column errors back to the LLM for self-correction.
4. **Query Caching**: Caching results to reduce LLM and BigQuery processing costs.

### 1.1 SQL Injection Prevention (`SQLValidator`)

In general enterprise applications, letting an LLM write and execute raw database queries poses a severe security vulnerability. A user might enter:
> "show me transactions; DROP TABLE fact_transactions;"

The `SQLValidator` class serves as our main defense. It parses and validates the generated SQL before sending it to BigQuery.

```python
class SQLValidator:
    BLOCKED_KEYWORDS = {
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
        "CREATE", "REPLACE", "MERGE", "GRANT", "REVOKE", "CALL",
    }

    @classmethod
    def validate(cls, sql: str, allowed_tables: set[str]) -> tuple[bool, str]:
        sql_upper = sql.upper().strip()

        # Rule 1: Force SELECT queries only
        if not sql_upper.startswith(("SELECT", "WITH")):
            return False, f"Only SELECT queries allowed. Got: {sql_upper[:20]}..."

        # Rule 2: Regex check for DML/DDL keywords as whole words
        for keyword in cls.BLOCKED_KEYWORDS:
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                return False, f"Blocked keyword detected: {keyword}"

        # Rule 3: Block multiple statements via semicolon injection
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        if len(statements) > 1:
            return False, "Multiple SQL statements not allowed"

        # Rule 4: Extract and monitor table references
        table_pattern = r'(?:FROM|JOIN)\s+`?([a-zA-Z0-9_.-]+)`?'
        referenced_tables = set(re.findall(table_pattern, sql, re.IGNORECASE))

        for table in referenced_tables:
            table_short = table.split(".")[-2] + "." + table.split(".")[-1] if table.count(".") >= 2 else table
            if table_short not in allowed_tables and table not in allowed_tables:
                logger.warning(f"Unknown table referenced: {table}")

        return True, ""
```

* **Why we use Regex boundaries (`\b`)**: Using a simple substring match like `"UPDATE" in sql` would block valid column names like `update_date` or `last_updated_timestamp`. The boundary check `\bUPDATE\b` guarantees that only the actual SQL statement keyword is blocked.
* **Why we parse tables**: BigQuery service accounts should be configured with read-only access (defense-in-depth), but logging and warning about tables not in our metadata index allows us to monitor for data leaks or unusual querying patterns.

---

### 1.2 Self-Healing SQL Execution Loop

If an LLM generates SQL that contains a minor syntax error, a missing partition filter, or a wrong column name, a standard system will crash. The `RAGQueryEngine` implements a self-healing loop:

```python
        for attempt in range(max_retries + 1):
            generated_sql = self._call_llm(prompt)
            generated_sql = self._clean_sql(generated_sql)

            # 1. Safety validation check
            is_valid, error = SQLValidator.validate(generated_sql, self.allowed_tables)
            if not is_valid:
                logger.warning(f"SQL validation failed: {error}")
                if attempt < max_retries:
                    # Feed the validation error back to the prompt
                    prompt = self._build_retry_prompt(question, context, generated_sql, error)
                    retry_count += 1
                    continue
                else:
                    return QueryResult(...)

            # 2. BigQuery execution check
            try:
                if self.mock_mode:
                    logger.info(f"[MOCK BQ] Executing RAG SQL query: {generated_sql[:100]}...")
                    raw_results = [{"customer_id": "cust_8271", "cnt": 12, "transaction_type": "UPI"}]
                else:
                    job = self.bq_client.query(generated_sql)
                    raw_results = [dict(row) for row in job.result()][:100]
                break  # Success!
            except Exception as e:
                logger.warning(f"SQL execution failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    # Feed execution engine error back to LLM
                    prompt = self._build_retry_prompt(question, context, generated_sql, str(e))
                    retry_count += 1
                else:
                    return QueryResult(...)
```

* **The logic**: If BigQuery throws an error (e.g. `Name customer_name not found inside dim_customer`), the engine formats this exact trace into a `retry_prompt` and sends it back to the LLM. The LLM sees the original schema, the failed query, and the exact compiler error, enabling it to write a correct replacement query immediately.

---

## 2. Autonomous Investigation Agent (`src/ai/agent.py`)

The `BankingInvestigationAgent` implements a classic **ReAct** (Reasoning and Acting) loop. Unlike the Query Engine, which executes one predefined sequence of steps, the agent receives a high-level goal, decides what tools to call, inspects the tools' outputs, and loops until it can draw a final conclusion.

### 2.1 ReAct Loop & History Context

At each step, the agent's prompt contains the entire history of its actions, thoughts, and observations so it can maintain context:

```python
        for step_num in range(1, self.max_steps + 1):
            step_start = time.time()

            # Format previous steps to preserve context within token limit
            history = [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "tool": s.tool,
                    "params": s.params,
                    "result_preview": s.result[:400] + ("..." if len(s.result) > 400 else "")
                }
                for s in steps
            ]
            
            prompt = AGENT_REACT_PROMPT.format(
                objective=objective,
                schemas=schemas_text,
                history=json.dumps(history, indent=2) if history else "None yet — first step"
            )

            raw_response = self._llm(prompt)
            action = self.parser.parse(raw_response)
```

---

### 2.2 Loop Detection Safeguards

LLM agents are notorious for getting stuck in infinite loops (e.g. running the same SQL query repeatedly and getting the same result, but refusing to change their plan). The `LoopDetector` class calculates a cryptographic hash of the `(tool + parameters)` to catch repeated calls:

```python
class LoopDetector:
    def __init__(self, window: int = 3):
        self.window = window
        self._recent: list[str] = []

    def check(self, tool: str, params: dict) -> bool:
        # Hash the tool name and sorted parameters
        key = hashlib.md5(
            (tool + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()

        # If this exact action appears twice in the tracking window, it is a loop
        is_loop = self._recent.count(key) >= 2
        self._recent.append(key)
        if len(self._recent) > self.window * 2:
            self._recent.pop(0)

        return is_loop
```

If a loop is detected, the agent loop terminates gracefully rather than wasting API credits:

```python
            if self.loop_detector.check(tool_name, params):
                logger.warning(f"Loop detected at step {step_num}: {tool_name} repeated")
                steps.append(AgentStep(
                    step_number=step_num,
                    thought=action.get("thought", ""),
                    tool=tool_name,
                    params=params,
                    result="LOOP_DETECTED: Breaking to prevent infinite repetition",
                    reasoning="Loop breaker",
                    duration_ms=(time.time() - step_start) * 1000,
                    success=False,
                ))
                break
```

---

### 2.3 Strict Parameter Verification

LLMs frequently struggle to match functions' precise python signatures. The `ToolRegistry` dynamically executes tool calls using standard python reflections while raising clear, clean exception messages:

```python
class ToolRegistry:
    def __init__(self, bq_client: bigquery.Client, project_id: str):
        self.bq = bq_client
        self.project_id = project_id
        self.mock_mode = bq_client is None
        self._tools: dict[str, Callable] = {}
        self._call_counts: dict[str, int] = {}

        # Register tools
        self.register("run_sql", self._run_sql)
        self.register("get_schema", self._get_schema)
        self.register("alert", self._send_alert)

    def call(self, name: str, params: dict) -> str:
        if name not in self._tools:
            return f"ERROR: Unknown tool '{name}'. Available: {list(self._tools.keys())}"
        self._call_counts[name] += 1
        try:
            # Reflection based tool execution
            return self._tools[name](**params)
        except TypeError as e:
            return f"ERROR: Wrong params for {name}: {e}"
        except Exception as e:
            return f"ERROR: {name} failed: {e}"
```

* **Why reflection matters**: If the LLM generates `params = {"query": "SELECT 1", "extra": "garbage"}`, python's `**kwargs` assignment throws a `TypeError`. The registry catches this and reports the parameters mismatch back to the agent reasoning loop, allowing it to self-correct its tool arguments on the next step.

---

## 3. Data Quality Engine (`src/quality/data_quality_engine.py`)

Data quality is the most critical pipeline step in financial data systems. The `DataQualityEngine` runs validation queries directly on BigQuery before downstream reporting tables are loaded.

### 3.1 Z-Score Anomaly Check

Volume spikes (duplicate rows) and volume drops (broken API ingest) are caught using a rolling Z-score over the last 30 days:

```python
    def check_volume_anomaly(self, table: str, z_threshold: float = 3.0):
        # Calculate daily count, average, standard deviation, and Z-score
        query = f"""
        WITH daily AS (
            SELECT DATE(_ingestion_ts) as dt, COUNT(*) as cnt
            FROM `{table}`
            WHERE DATE(_ingestion_ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            GROUP BY 1
        ), stats AS (
            SELECT AVG(cnt) as mu, STDDEV(cnt) as sigma FROM daily WHERE dt < CURRENT_DATE()
        )
        SELECT d.cnt as today, s.mu, s.sigma,
               ABS(d.cnt - s.mu) / NULLIF(s.sigma, 0) as z
        FROM daily d, stats s WHERE d.dt = CURRENT_DATE()
        """
        # Execute query
        row = list(self.client.query(query).result())[0]
        z = row.z or 0
        passed = z <= z_threshold
        
        self.results.append(CheckResult(
            check_name="volume_anomaly", table=table, passed=passed,
            severity=Severity.WARNING, value=z, threshold=z_threshold,
            message=f"Today: {row.today} rows (mean: {row.mu:.0f}, z: {z:.2f})"
        ))
```

* **Why Z-Score**: Rather than hardcoding static limits (e.g. "raise alert if row count < 10000"), which break as the bank grows, Z-Score measures deviation dynamically. If yesterday was within normal variance but today deviates by more than `3` standard deviations from the 30-day mean, it is statistically marked as an anomaly.
