# AI/ML Concepts Guide — Banking AI Platform

> A practical reference for understanding how AI powers this banking platform. Covers embeddings, transformers, RAG, agentic AI, and prompting — with banking-specific examples.

---

## 1. The Big Picture

```
Human Language → [AI Magic Box] → Actionable Banking Insights

But what's inside the magic box?

Text ("show me fraud alerts")
  → Tokenization (split into subword pieces)
    → Static Embeddings (each token → 1536-dim number vector)
      → Positional Encoding (capture word order)
        → Self-Attention (each token understands context from others)
          → Contextual Embeddings (same word, different meaning based on context)
            → LM Head (predict next token)
              → Output ("SELECT alert_type, COUNT(*)...")
```

---

## 2. Embeddings — Text as Numbers

### What Are Embeddings?
Machines don't understand words. Embeddings convert text into **vectors** (arrays of numbers) that capture semantic meaning. Similar meanings → similar vectors.

### Static Embeddings (Word2Vec, ~2013)
Each word gets a **fixed** 300-dimensional vector regardless of context.

```
cat → [0.8, 0.1, 0.3, ..., 1.0]  (300 numbers)
dog → [0.9, 0.2, 0.5, ..., 0.8]  (300 numbers)

300 dimensions = 300 features (has_tail, weight, color, is_animal, ...)
Each feature gets a score between -1 and 1.
```

**Training:** Models like Word2Vec are trained on massive text corpora (Wikipedia). They learn word meanings by analyzing which words appear near each other. Two architectures:
- **CBOW** (Continuous Bag of Words): Predict a word from its neighbors
- **Skip-gram**: Predict neighbors from a word

**Limitation:** "bank" (river bank) and "bank" (financial institution) get the SAME vector. Context is lost.

### Contextual Embeddings (Transformers, 2017+)
Each word gets a **different** vector depending on the surrounding sentence. Modern models use **1536+ dimensions** (vs Word2Vec's 300).

```
"I deposited money in the bank" → bank = [financial context vector]
"I sat by the river bank"       → bank = [nature context vector]

Same word, DIFFERENT embeddings. This is the breakthrough.
```

**How?** Self-attention mechanism in transformers. Each word "attends to" every other word in the sentence, learning relationships.

### Cosine Similarity — Measuring Meaning Distance

```
similarity = dot(A, B) / (||A|| × ||B||)

Why cosine (not Euclidean distance)?
- Cosine measures DIRECTION (meaning), not magnitude (length)
- A short document and long document about "fraud" point the same direction
- Euclidean would say they're far apart (different lengths)
- Cosine correctly says they're similar (same topic)

Scale:
 1.0 = identical meaning
 0.0 = completely unrelated
-1.0 = opposite meaning

Banking example:
- "suspicious transaction" ↔ "fraud_score column" → similarity: 0.89 ✅
- "suspicious transaction" ↔ "customer_name column" → similarity: 0.12 ❌
```

---

## 3. Transformers — The Architecture Behind Everything

### What Are Transformers?
The neural network architecture powering ChatGPT, Gemini, Claude, and every modern LLM. Introduced in the 2017 paper **"Attention Is All You Need"**.

### Three Parts of a Transformer

| Part | What It Does | Details |
|------|-------------|---------|
| **Tokenizer** | Text → tokens → static embeddings + positions | Splits text, assigns IDs, looks up embeddings, adds position info |
| **Block Stack** | Static → contextual embeddings | Multiple self-attention layers that add context to each token |
| **LM Head** | Predicts the next token | Takes the last token's contextual embedding, outputs probability distribution |

### Step-by-Step: How ChatGPT Generates a Response

```
Input: "What is the fraud rate today?"

1. TOKENIZE:
   Tokens: ["What", "is", "the", "fraud", "rate", "today", "?"]
   Token IDs: [1724, 374, 290, 8924, 4478, 3432, 30]
   
2. STATIC EMBEDDINGS:
   Each token ID → lookup 1536-dim vector from vocabulary
   "fraud" → [0.23, -0.45, 0.89, ...]  (same vector every time — static)
   
3. POSITION ENCODING:
   Add position information: "fraud" is at position 3
   Why? "cats love dogs" ≠ "dogs love cats" — same words, different meaning
   
4. TRANSFORMER BLOCKS (self-attention):
   Each token attends to other tokens to build context:
   "fraud" attends to "rate" and "today" → understands we want fraud RATE for TODAY
   Static embedding → Contextual embedding (now has full sentence context)
   
5. LM HEAD:
   Takes LAST token's contextual embedding ("?" has context of entire question)
   Predicts next most likely token → "The"
   
6. REPEAT:
   "What is the fraud rate today? The" → predict → "current"
   "What is the fraud rate today? The current" → predict → "fraud"
   ... token by token until done
```

### Self-Attention Types

| Type | How It Works | Used In | Banking Use |
|------|-------------|---------|-------------|
| **Masked** | Each token sees ONLY previous tokens | GPT (decoder-only) | SQL generation, text generation |
| **Bidirectional** | Each token sees ALL tokens | BERT (encoder-only) | Document classification, embeddings |

### Encoder vs Decoder

| Model Type | Architecture | Good At | Banking Application |
|------------|-------------|---------|---------------------|
| **Encoder-only** (BERT) | Bidirectional attention | Understanding text | Classify compliance docs, generate embeddings |
| **Decoder-only** (GPT) | Masked attention | Generating text | Generate SQL, write reports, agent reasoning |
| **Encoder-Decoder** (T5) | Both | Translation/summarization | Summarize audit reports |

---

## 4. RAG — Retrieval-Augmented Generation

### The Problem
LLMs don't know your private banking data. Ask GPT "what's our fraud rate?" and it will hallucinate an answer.

### The Solution: RAG

```
DON'T: User → LLM → Hallucinated answer ❌
DO:    User → Retrieve real context → LLM + context → Grounded answer ✅

1. RETRIEVAL: Find relevant schemas using vector search (cosine similarity)
2. AUGMENTATION: Inject schemas into the LLM prompt as context
3. GENERATION: LLM generates SQL using ONLY the provided schemas
```

### Why RAG (not fine-tuning)?
| Factor | RAG | Fine-tuning |
|--------|-----|-------------|
| Data freshness | Always current (retrieves live schemas) | Stale (needs retraining) |
| Cost | Low (just API calls) | High (GPU training) |
| Flexibility | Add/change schemas instantly | Retrain for every change |
| Auditability | Can inspect retrieved context | Black box |
| **Banking verdict** | **✅ Production choice** | ❌ Too rigid |

---

## 5. Agentic AI — Autonomous Investigation

### What's an Agent?
An AI that takes an **objective** and autonomously decides what steps to take, what tools to use, and when to stop. Unlike RAG (1 question → 1 answer), agents handle multi-step investigations.

### ReAct Pattern (Reasoning + Acting)
```
OBJECTIVE: "Investigate why fraud alerts tripled yesterday"

Step 1:
  THOUGHT: "First, I need to see yesterday's fraud alert count vs normal"
  ACTION: run_sql("SELECT COUNT(*) FROM fact_fraud_alerts WHERE created_date = ...")
  OBSERVATION: "450 alerts vs 150 average"

Step 2:
  THOUGHT: "3x spike confirmed. Which alert types increased most?"
  ACTION: run_sql("SELECT alert_type, COUNT(*) FROM fact_fraud_alerts WHERE ...")
  OBSERVATION: "VELOCITY alerts: 320 (normally 50)"

Step 3:
  THOUGHT: "Velocity alerts surged. Are they from specific accounts?"
  ACTION: run_sql("SELECT customer_id, COUNT(*) FROM fact_fraud_alerts WHERE alert_type='VELOCITY'...")
  OBSERVATION: "Top 3 accounts generated 280 of 320 alerts"

Step 4:
  THOUGHT: "3 accounts causing 87% of alerts. Likely coordinated fraud or false positive from system change"
  ACTION: alert("3 accounts generating 87% of velocity alerts — investigate for coordinated fraud", "critical")

Step 5:
  THOUGHT: "Investigation complete — root cause identified"
  TOOL: DONE
  SUMMARY: "Fraud alerts tripled due to 3 accounts triggering 280 velocity alerts. Recommend: freeze accounts, investigate for coordinated fraud attempt."
```

---

## 6. Prompting Techniques

| Technique | When to Use | Banking Example |
|-----------|-------------|-----------------|
| **Zero-shot** | Simple, straightforward queries | "How many transactions today?" |
| **Few-shot** | Complex patterns the LLM might not know | "Generate SQL with banking-specific JOIN patterns" |
| **Chain-of-thought** | Multi-step reasoning | "Think step by step about which tables to join" |
| **ReAct** | Autonomous multi-step tasks | Agent investigating fraud patterns |
| **System prompt** | Every LLM call | "You are a BigQuery expert for a banking platform" |

---

## 7. Tokenization Deep Dive

Tokens are **model-specific** — the same text creates different tokens on different models:

```
Input: "10kilometers"

GPT-4:   ["10", "k", "ms"]      → 3 tokens
GPT-3.5: ["10", "kil", "ometers"] → 3 tokens (different split!)
DaVinci: ["10", "km", "s"]      → 3 tokens (yet another split!)

Why this matters:
- APIs charge PER TOKEN (GPT-4: ~$0.03/1K tokens)
- Context window is measured in tokens (GPT-4: 128K tokens)
- More efficient tokenization = more content in context window
```

### Token IDs
Each token maps to a numeric ID in the model's vocabulary:
```
"I"    → Token ID: 40
"want" → Token ID: 1682 (GPT-4) or 765 (DaVinci) — different per model!
```

These IDs are used internally to look up static embeddings from the vocabulary table. The ID itself has no semantic meaning — it's just a lookup key.

---

## 8. Quick Reference

| Concept | One-Line Explanation |
|---------|---------------------|
| **Embedding** | Text → numbers that capture meaning |
| **Static embedding** | Same word = same vector always (Word2Vec) |
| **Contextual embedding** | Same word = different vector based on context (Transformer) |
| **Cosine similarity** | Measure how similar two vectors are (0=unrelated, 1=identical) |
| **Tokenization** | Split text into subword pieces for the model |
| **Self-attention** | Each token looks at all others to understand relationships |
| **Masked attention** | Token can only look at PREVIOUS tokens (GPT-style) |
| **Transformer** | Neural net with attention — powers all modern AI |
| **Encoder** | Understands text (BERT) — good for embeddings, classification |
| **Decoder** | Generates text (GPT) — good for SQL, reports, conversations |
| **LM Head** | Final layer that predicts the next token |
| **RAG** | Retrieve context + augment prompt + generate answer |
| **Vector store** | Database of embeddings for semantic search |
| **Agent** | Autonomous AI that plans and executes multi-step tasks |
| **ReAct** | Reasoning + Acting pattern for agents |
| **LLM** | Large Language Model — a big transformer trained on massive text data |
