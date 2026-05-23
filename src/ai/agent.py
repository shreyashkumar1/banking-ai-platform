"""
BankingInvestigationAgent — Autonomous multi-step fraud and anomaly investigation.

A real ReAct agent loop:
- Builds a prompt with objective + full step history
- Parses structured JSON actions from the LLM response
- Executes tools (SQL, schema lookup, alerts)
- Detects and breaks loops (same SQL twice = stuck)
- Caps cost at max_steps * cost_per_step
- Produces a full audit trail (required for banking compliance)
"""

from google.cloud import bigquery
from google.auth.exceptions import DefaultCredentialsError
from dataclasses import dataclass, field
from typing import Optional, Callable
import json
import logging
import time
import hashlib

from src.ai.prompts import AGENT_REACT_PROMPT
from src.ai.vector_store import BANKING_SCHEMAS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════

@dataclass
class AgentStep:
    step_number: int
    thought: str
    tool: str
    params: dict
    result: str
    reasoning: str
    duration_ms: float
    success: bool = True


@dataclass
class AgentResult:
    objective: str
    steps: list[AgentStep]
    summary: str
    root_cause: str
    recommendation: str
    total_duration_ms: float
    total_queries: int
    completed: bool          # False if hit max_steps without DONE
    cost_estimate_usd: float


# ══════════════════════════════════════════════════════════════════
# Tool Registry — All tools the agent can call
# ══════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Manages agent tools with input validation and error handling.
    
    Adding a new tool: just add a method and register it.
    The agent prompt is automatically updated with available tool names.
    """

    def __init__(self, bq_client: bigquery.Client, project_id: str):
        self.bq = bq_client
        self.project_id = project_id
        self.mock_mode = bq_client is None
        self._tools: dict[str, Callable] = {}
        self._call_counts: dict[str, int] = {}

        # Register built-in tools
        self.register("run_sql", self._run_sql)
        self.register("get_schema", self._get_schema)
        self.register("alert", self._send_alert)

    def register(self, name: str, fn: Callable):
        self._tools[name] = fn
        self._call_counts[name] = 0

    def call(self, name: str, params: dict) -> str:
        if name not in self._tools:
            return f"ERROR: Unknown tool '{name}'. Available: {list(self._tools.keys())}"
        self._call_counts[name] += 1
        try:
            return self._tools[name](**params)
        except TypeError as e:
            return f"ERROR: Wrong params for {name}: {e}"
        except Exception as e:
            return f"ERROR: {name} failed: {e}"

    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    def call_counts(self) -> dict[str, int]:
        return dict(self._call_counts)

    # ── Built-in tools ──

    def _run_sql(self, query: str, max_rows: int = 50) -> str:
        """Execute BigQuery SQL. Returns JSON string of results."""
        # Safety: agent can only run SELECT
        upper = query.strip().upper()
        if not upper.startswith(("SELECT", "WITH")):
            return "ERROR: Agent can only run SELECT queries"

        if self.mock_mode:
            logger.info(f"[MOCK BQ] Agent executing SQL: {query[:100]}...")
            if "fact_fraud_alerts" in query.lower():
                if "count" in query.lower():
                    return json.dumps([{"risk_level": "HIGH", "cnt": 47, "avg_score": 0.89}])
                return json.dumps([
                    {"customer_id": "cust_8271", "alert_count": 8, "max_score": 0.94, "risk_category": "HIGH"},
                    {"customer_id": "cust_1982", "alert_count": 5, "max_score": 0.91, "risk_category": "MEDIUM"},
                ])
            elif "fact_transactions" in query.lower():
                return json.dumps([
                    {"transaction_type": "UPI", "channel": "MOBILE", "cnt": 18, "total_inr": 12500},
                    {"transaction_type": "TRANSFER", "channel": "WEB", "cnt": 2, "total_inr": 45000},
                ])
            return json.dumps([{"status": "OK", "details": "Mock SQL results (Mock Mode)"}])

        try:
            job = self.bq.query(query)
            rows = [dict(row) for row in job.result()][:max_rows]
            if not rows:
                return "[]  (query returned 0 rows)"
            return json.dumps(rows, default=str, indent=2)
        except Exception as e:
            return f"BigQuery error: {e}"

    def _get_schema(self, table: str) -> str:
        """Return column names and types for a BigQuery table."""
        if self.mock_mode:
            from src.ai.vector_store import BANKING_SCHEMAS
            # Find the schema
            for name, info in BANKING_SCHEMAS.items():
                if table in name:
                    return json.dumps({c["name"]: {"type": c["type"], "mode": "NULLABLE"} for c in info["columns"]}, indent=2)
            return json.dumps({"error": f"Table {table} not found in mock schemas"}, indent=2)

        try:
            t = self.bq.get_table(f"{self.project_id}.{table}")
            schema = {f.name: {"type": f.field_type, "mode": f.mode} for f in t.schema}
            return json.dumps(schema, indent=2)
        except Exception as e:
            return f"Schema error: {e}"

    def _send_alert(self, message: str, severity: str = "info") -> str:
        """Send alert to team channel (Slack/PagerDuty in production)."""
        valid_severities = {"info", "warning", "critical"}
        if severity not in valid_severities:
            severity = "info"
        logger.warning(f"[AGENT ALERT:{severity.upper()}] {message}")
        # Production: post to Slack webhook / PagerDuty API
        return f"Alert dispatched (severity={severity}): {message}"


# ══════════════════════════════════════════════════════════════════
# LLM Response Parser — Extract structured action from LLM output
# ══════════════════════════════════════════════════════════════════

class ActionParser:
    """Parse LLM JSON responses into agent actions.
    
    LLMs sometimes wrap JSON in markdown, add trailing commas, etc.
    This parser handles common failure modes gracefully.
    """

    @staticmethod
    def parse(raw_response: str) -> dict:
        """Parse LLM response into action dict."""
        # Strip markdown code blocks
        text = raw_response.strip()
        for marker in ["```json", "```JSON", "```"]:
            if text.startswith(marker):
                text = text[len(marker):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try direct JSON parse
        try:
            action = json.loads(text)
            return ActionParser._validate(action)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from surrounding text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                action = json.loads(text[start:end])
                return ActionParser._validate(action)
            except json.JSONDecodeError:
                pass

        # Complete failure — return a safe fallback
        logger.error(f"Failed to parse LLM action: {raw_response[:200]}")
        return {
            "thought": "Failed to parse LLM response",
            "tool": "DONE",
            "params": {},
            "summary": f"Investigation halted: LLM returned unparseable response.",
            "root_cause": "LLM parsing failure",
            "recommendation": "Retry or investigate manually",
        }

    @staticmethod
    def _validate(action: dict) -> dict:
        """Ensure required fields exist."""
        action.setdefault("thought", "")
        action.setdefault("tool", "DONE")
        action.setdefault("params", {})
        action.setdefault("reasoning", "")
        return action


# ══════════════════════════════════════════════════════════════════
# Loop Detector — Prevent infinite repetition
# ══════════════════════════════════════════════════════════════════

class LoopDetector:
    """Detect if agent is repeating itself.
    
    Two failure modes:
    1. Same tool + same params repeated → agent is stuck
    2. No progress toward DONE after N steps → agent is lost
    """

    def __init__(self, window: int = 3):
        self.window = window
        self._recent: list[str] = []

    def check(self, tool: str, params: dict) -> bool:
        """Returns True if this action looks like a loop."""
        key = hashlib.md5(
            (tool + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()

        is_loop = self._recent.count(key) >= 2
        self._recent.append(key)
        if len(self._recent) > self.window * 2:
            self._recent.pop(0)

        return is_loop


# ══════════════════════════════════════════════════════════════════
# Main Agent
# ══════════════════════════════════════════════════════════════════

class BankingInvestigationAgent:
    """Autonomous banking investigation agent with ReAct reasoning loop.
    
    Cost model:
    - Each LLM call: ~$0.05 (GPT-4 at ~2K tokens per call)
    - max_steps=10 → $0.50 max per investigation
    - BigQuery SQL: ~$0.005 per TB scanned (negligible for targeted queries)
    
    Audit trail:
    - Every step logged with thought, tool, params, result, duration
    - Required for banking compliance (Basel III operational risk logs)
    - Stored to BigQuery governance.agent_audit_log in production
    """

    COST_PER_STEP_USD = 0.05

    def __init__(self, project_id: str, max_steps: int = 10,
                 llm_fn: Optional[Callable[[str], str]] = None):
        """
        Args:
            project_id: GCP project ID
            max_steps: Hard cap on investigation steps (cost control)
            llm_fn: LLM callable — inject real LLM in production,
                    mock in tests. Signature: (prompt: str) -> str
        """
        self.project_id = project_id
        self.max_steps = max_steps
        try:
            self.bq = bigquery.Client(project=project_id)
            self.mock_mode = False
        except (DefaultCredentialsError, Exception) as e:
            logger.warning(f"Could not initialize BigQuery client: {e}. Running in MOCK mode.")
            self.bq = None
            self.mock_mode = True

        self.tools = ToolRegistry(self.bq, project_id)
        self.loop_detector = LoopDetector()
        self.parser = ActionParser()

        # Dependency-injected LLM — mock by default, real in production
        self._llm = llm_fn or self._mock_llm

        logger.info(
            f"BankingInvestigationAgent ready: "
            f"project={project_id}, max_steps={max_steps}, "
            f"tools={self.tools.available_tools()}"
        )

    def investigate(self, objective: str) -> AgentResult:
        """Run autonomous investigation using ReAct reasoning loop.
        
        The loop:
        1. Build prompt: objective + schema context + full step history
        2. Call LLM → get JSON action (thought + tool + params)
        3. Check for loops (same action repeated → break)
        4. Execute tool → get result
        5. Append step to history
        6. If tool == DONE → return result
        7. If step_num == max_steps → return with incomplete flag
        """
        start_time = time.time()
        steps: list[AgentStep] = []
        total_queries = 0

        logger.info(f"Investigation started: '{objective}'")

        for step_num in range(1, self.max_steps + 1):
            step_start = time.time()

            # Build history for prompt (truncate result text to save tokens)
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

            # Build ReAct prompt
            schemas_text = "\n".join(
                f"- {name}: {info['description']}"
                for name, info in BANKING_SCHEMAS.items()
            )
            prompt = AGENT_REACT_PROMPT.format(
                objective=objective,
                schemas=schemas_text,
                history=json.dumps(history, indent=2) if history else "None yet — first step"
            )

            # Call LLM
            raw_response = self._llm(prompt)
            action = self.parser.parse(raw_response)

            tool_name = action.get("tool", "DONE")
            params = action.get("params", {})

            # ── Terminal condition ──
            if tool_name == "DONE":
                total_duration = (time.time() - start_time) * 1000
                logger.info(f"Investigation DONE in {step_num} steps ({total_duration:.0f}ms)")
                return AgentResult(
                    objective=objective,
                    steps=steps,
                    summary=action.get("summary", "Investigation complete"),
                    root_cause=action.get("root_cause", "See summary"),
                    recommendation=action.get("recommendation", "See summary"),
                    total_duration_ms=total_duration,
                    total_queries=total_queries,
                    completed=True,
                    cost_estimate_usd=step_num * self.COST_PER_STEP_USD,
                )

            # ── Loop detection ──
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

            # ── Execute tool ──
            result = self.tools.call(tool_name, params)
            if tool_name == "run_sql":
                total_queries += 1

            step_duration = (time.time() - step_start) * 1000

            steps.append(AgentStep(
                step_number=step_num,
                thought=action.get("thought", ""),
                tool=tool_name,
                params=params,
                result=result,
                reasoning=action.get("reasoning", ""),
                duration_ms=step_duration,
                success=not result.startswith("ERROR"),
            ))

            logger.info(
                f"Step {step_num}/{self.max_steps}: {tool_name} → "
                f"{'OK' if not result.startswith('ERROR') else 'FAIL'} "
                f"({step_duration:.0f}ms)"
            )

        # Max steps reached without DONE
        total_duration = (time.time() - start_time) * 1000
        logger.warning(f"Max steps ({self.max_steps}) reached for: '{objective}'")

        return AgentResult(
            objective=objective,
            steps=steps,
            summary=(
                f"Investigation incomplete after {self.max_steps} steps. "
                f"Last action: {steps[-1].tool if steps else 'none'}. "
                f"Manual review recommended."
            ),
            root_cause="Investigation incomplete — max steps reached",
            recommendation="Review step history and continue investigation manually",
            total_duration_ms=total_duration,
            total_queries=total_queries,
            completed=False,
            cost_estimate_usd=self.max_steps * self.COST_PER_STEP_USD,
        )

    def get_audit_trail(self, result: AgentResult) -> dict:
        """Export structured audit trail for compliance logging."""
        return {
            "objective": result.objective,
            "completed": result.completed,
            "total_steps": len(result.steps),
            "total_queries": result.total_queries,
            "duration_ms": result.total_duration_ms,
            "cost_usd": result.cost_estimate_usd,
            "tool_usage": self.tools.call_counts(),
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "tool": s.tool,
                    "params": s.params,
                    "result_length": len(s.result),
                    "duration_ms": s.duration_ms,
                    "success": s.success,
                }
                for s in result.steps
            ],
            "summary": result.summary,
            "root_cause": result.root_cause,
            "recommendation": result.recommendation,
        }

    @staticmethod
    def _mock_llm(prompt: str) -> str:
        """Deterministic mock LLM for development/testing.
        
        Simulates a real investigation by cycling through realistic steps.
        Replace with real LLM client in production.
        """
        # Count steps based on how much history is in the prompt
        history_count = prompt.count('"step":')

        if history_count == 0:
            return json.dumps({
                "thought": "Starting investigation — first I need to understand the current alert volume",
                "tool": "run_sql",
                "params": {
                    "query": "SELECT risk_level, COUNT(*) as cnt, AVG(fraud_score) as avg_score FROM `analytics.fact_fraud_alerts` WHERE created_date = CURRENT_DATE() GROUP BY risk_level ORDER BY cnt DESC"
                },
                "reasoning": "Establish baseline: how many alerts and what risk distribution"
            })
        elif history_count == 1:
            return json.dumps({
                "thought": "Got alert counts. Now I need to see which customers are flagged",
                "tool": "run_sql",
                "params": {
                    "query": "SELECT fa.customer_id, COUNT(*) as alert_count, MAX(fa.fraud_score) as max_score, dc.risk_category FROM `analytics.fact_fraud_alerts` fa JOIN `analytics.dim_customer` dc ON fa.customer_id = dc.customer_id WHERE fa.created_date = CURRENT_DATE() GROUP BY fa.customer_id, dc.risk_category HAVING alert_count > 1 ORDER BY max_score DESC LIMIT 10"
                },
                "reasoning": "Multi-alert customers are highest priority — joining with customer risk profile"
            })
        elif history_count == 2:
            return json.dumps({
                "thought": "Found repeat-alert customers. Check transaction patterns for top suspect",
                "tool": "run_sql",
                "params": {
                    "query": "SELECT transaction_type, channel, COUNT(*) as cnt, SUM(amount) as total_inr FROM `analytics.fact_transactions` WHERE customer_id = (SELECT customer_id FROM `analytics.fact_fraud_alerts` WHERE created_date = CURRENT_DATE() GROUP BY customer_id ORDER BY COUNT(*) DESC LIMIT 1) AND created_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) GROUP BY transaction_type, channel"
                },
                "reasoning": "7-day transaction pattern for highest-risk customer"
            })
        else:
            return json.dumps({
                "thought": "Sufficient data collected to draw conclusions",
                "tool": "DONE",
                "params": {},
                "summary": "Fraud investigation complete. Identified 3 HIGH-risk customers with repeated alerts today, showing abnormal UPI transaction velocity (15-20 transactions/hour vs normal 2-3). Pattern matches velocity fraud signature.",
                "root_cause": "UPI velocity fraud: customers making rapid small-value transactions (₹499-999) to multiple payees — likely testing stolen account credentials",
                "recommendation": "1) Temporarily freeze UPI limits for flagged accounts pending manual review. 2) Alert fraud team via #fraud-alerts Slack channel. 3) Check if payees are in fraud merchant blacklist."
            })
