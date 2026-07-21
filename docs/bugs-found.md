# Bugs found while building this project

All three of these were real failures hit while actually running the code (not hypothetical edge cases), caught by tests or by the benchmark harness itself, and are now locked in as regression tests so they can't silently regress.

---

## 1. Mock ReAct output wasn't valid JSON

**Where:** `src/core/llm_client.py::MockLLMClient._complete_react_text`
**Caught by:** `tests/integration/test_agents.py::test_react_agent_answers_arithmetic` failing with a confusing `KeyError`
**Locked in by:** `tests/regression/test_mock_react_json_format.py`

### What happened

The mock's ReAct branch formatted the `Action Input:` line with an f-string directly over a Python dict:

```python
text = (
    f"Thought: I need more information to answer '{last_user_msg}'.\n"
    f"Action: {tool_use['name']}\n"
    f"Action Input: {tool_use['input']}"   # <-- str(dict), not JSON
)
```

For a calculator call this produced:

```
Action Input: {'expression': '6 * 7'}
```

That's valid **Python** repr (single quotes) but not valid **JSON**. `react_agent.py` parses `Action Input` with `json.loads`, which raised a `JSONDecodeError` on the single quotes. That exception was caught by a bare `except Exception: args = {}` a few lines later — silently swallowing the parse failure — so the agent went on to call `calculator.run(**{})` with no arguments at all, which raised `KeyError: 'expression'` inside the tool, surfaced to the user as:

```
final_answer = "[mock-mode] Based on the retrieved evidence: Observation: Tool error: 'expression'"
```

A confusing failure two layers removed from the actual bug (bad JSON formatting), because the exception-swallowing at the parsing layer hid where things actually went wrong.

### The fix

Use `json.dumps` instead of `str()`/f-string interpolation on the dict:

```python
text = (
    f"Thought: I need more information to answer '{last_user_msg}'.\n"
    f"Action: {tool_use['name']}\n"
    f"Action Input: {json.dumps(tool_use['input'])}"
)
```

### Lesson

This is the same category of lesson noted in `chatbot-rag`: a mock LLM is still a text-generation contract with the rest of the system, and it needs to be precise about that contract (valid JSON where JSON is expected), not just "close enough for a demo." The bare `except: args = {}` in the parser also deserves scrutiny in its own right — it turned a clear parsing bug into a much more confusing downstream `KeyError`.

---

## 2. Mock calculator only extracted the first two-number pair

**Where:** `src/core/llm_client.py::MockLLMClient` (originally `_MATH_PATTERN`)
**Caught by:** `python scripts/run_benchmark.py` reporting an 80% pass rate instead of the expected 100% on a deterministic arithmetic example
**Locked in by:** `tests/regression/test_mock_calculator_expression_extraction.py`

### What happened

The original regex used to detect and extract an arithmetic expression from a user query was:

```python
_MATH_PATTERN = re.compile(r"\d+\s*[\+\-\*/x×]\s*\d+")
```

This matches exactly one operator between exactly two numbers. For the eval example `"What is (25 + 15) / 2?"`, `.search()` returns only the **first** match: `"25 + 15"` — silently dropping the `(...) / 2` part. The calculator then correctly computed `25 + 15 = 40`, which is the right answer to the wrong (truncated) expression; the query actually asks for `(25 + 15) / 2 = 20`.

Because the tool executed without raising any error, nothing crashed — the benchmark's pass-rate metric was what actually surfaced the problem: it flagged a 20% failure rate on a task category (arithmetic) that should be 100% deterministic and gave a wrong-not-error result, which is a more dangerous failure mode than a crash.

### The fix

Replaced the fixed two-operand pattern with a maximal-run extraction that accepts parentheses, decimals, and multiple operators, then validates the match actually contains at least one digit and one operator before treating it as an expression:

```python
_MATH_CHARS_PATTERN = re.compile(r"[\d\.\s\+\-\*/x×\(\)]{3,}")

def _extract_math_expression(self, lowered_text: str) -> str | None:
    best_match = None
    for match in self._MATH_CHARS_PATTERN.finditer(lowered_text):
        candidate = match.group(0).strip()
        has_digit = any(c.isdigit() for c in candidate)
        has_operator = any(c in "+-*/x×" for c in candidate)
        if has_digit and has_operator and (best_match is None or len(candidate) > len(best_match)):
            best_match = candidate
    ...
```

### Lesson

A quantitative benchmark caught a bug that manual spot-checking (a couple of `"12 * 8"`-style test cases) had missed, because the bug only shows up on expressions structurally different from the simplest case (parentheses + a second operator). This is the concrete payoff of "run metrics and benchmarks over qualitative claims" — the wrong-but-plausible-looking answer (40 instead of 20) is exactly the kind of failure that's easy to miss by eye and easy to catch by comparing against a known ground truth at scale.

---

## 3. Reranker's missing-dependency fallback didn't actually fire

**Where:** `src/tools/registry.py::build_default_registry` (added when introducing the cross-encoder reranker, `src/retrieval/reranker.py`)
**Caught by:** running `python scripts/run_benchmark.py` in an environment without `sentence-transformers` installed — crashed mid-run instead of falling back
**Locked in by:** `tests/regression/test_reranker_missing_dependency_fallback.py`

### What happened

The first version of `build_default_registry()` wrapped only the *construction* of `RerankedRetriever` in a `try/except ImportError`:

```python
try:
    from src.retrieval.reranker import CrossEncoderReranker, RerankedRetriever
    retriever = RerankedRetriever(base=retriever, reranker=CrossEncoderReranker(), ...)
except ImportError:
    ...  # fall back to hybrid-only
```

But `CrossEncoderReranker` imports `sentence_transformers` **lazily**, inside `_load()`, which only runs the first time `.rerank()` is actually called — not when the object is constructed. So building the registry always succeeded even without the dependency installed, and the `ModuleNotFoundError` instead surfaced several calls later, deep inside an agent's `run()` loop, the first time it tried to retrieve anything:

```
ModuleNotFoundError: No module named 'sentence_transformers'
  File ".../src/retrieval/reranker.py", line 55, in rerank
    model = self._load()
```

Caught immediately by actually running `scripts/run_benchmark.py` end to end in a clean environment rather than only unit-testing the registry construction in isolation.

### The fix

Two layers, matching the "defense in depth" pattern already used elsewhere in this codebase:

1. `registry.py` now does a cheap top-level `import sentence_transformers` check *before* deciding whether to wrap the retriever, so the `try/except` actually covers the thing that can fail.
2. `CrossEncoderReranker.rerank()` also catches `ImportError` around its own lazy load and falls back to returning the un-reranked candidates, so anyone constructing a `RerankedRetriever` directly (bypassing the registry) still degrades gracefully instead of crashing.

### Lesson

A `try/except` only protects the code that's actually inside the `try` block — wrapping the *constructor* of an object that does lazy imports doesn't wrap the *later* call that triggers the import. The same class of bug as #1 above (an exception swallowed or missed at the wrong layer), just the opposite direction: here nothing was swallowing it, the `except` simply wasn't positioned around the code that could raise it.

---

## 4. pgvector `load_from_store()` raised the wrong exception type on a fresh database

**Where:** `src/retrieval/hybrid_pgvector.py::HybridPGVectorRetriever.load_from_store` (added when swapping the pickle-file vector index for a real Postgres/pgvector backend)
**Caught by:** running `indexer.load_or_build_index()` against a genuinely empty Postgres database (first deploy) instead of only testing against one that already had the table
**Locked in by:** `tests/regression/test_pgvector_fresh_database_fallback.py`

### What happened

`indexer.py::load_or_build_index()` is written to *try loading first, and only build a fresh index if that raises `RuntimeError`* -- the same "just works on a clean clone" pattern the original pickle-file path already had. `load_from_store()` was supposed to raise `RuntimeError("No chunks found...")` when the table is empty so that fallback fires. But on a **brand-new** database, the `document_chunks` table doesn't exist yet at all, and the `SELECT` inside `fetch_all()` failed with:

```
sqlalchemy.exc.ProgrammingError: relation "document_chunks" does not exist
```

-- a completely different exception type than the `RuntimeError` the caller's `except` clause was watching for, so it propagated straight up and crashed the first API request on a fresh deployment instead of falling back to `build_pgvector_index()` like it should have.

Caught by deliberately dropping the table and re-running the load path against the live Postgres instance in this environment, not by unit-testing `load_from_store()` in isolation against a database that already happened to have the table from a previous test run.

### The fix

`load_from_store()` now calls `store.ensure_schema()` (idempotent `CREATE TABLE IF NOT EXISTS`) *before* `fetch_all()`, so a fresh database just returns an empty list -- which correctly raises the `RuntimeError` the caller already knows how to catch, instead of a raw database exception.

### Lesson

Same root cause as bug #3, different direction: that one was a missing dependency raising an uncaught exception type; this one is a missing *table* doing the same thing. Both are instances of "the code path that only runs once, on a machine's very first startup, is the path least likely to get exercised by normal development" -- which is exactly why it's worth deliberately testing against a genuinely fresh environment (empty DB, missing package) rather than only ever running tests against a machine that's already been set up once.
