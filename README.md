# LossLens – Tool-Calling LLM Agent for Censored Severity Modeling

Dependency-free (Python 3.9+ standard library) agent framework that lets an LLM investigate an
insurance claims table: validate data, retrieve rating rules, fit a **right-censored** severity
model, select between models, and price an excess layer with a bootstrap confidence interval.
**Synthetic data only.**

## Components
- `agent/core.py` – agent loop: tool schemas, tool-use/tool-result turns, errors returned to the model,
  hard step budget, JSONL audit log.
- `agent/tools.py` – 7 tools: guarded SQL (SELECT-only, single statement, read-only connection, 50-row cap),
  data-quality report, censored 2-component mixed-exponential EM fit, AIC/BIC model comparison,
  layer pricing vs an empirical check on fully exposed claims, bootstrap layer CI, rule retrieval.
- `agent/retrieval.py` – BM25 over 14 synthetic rating-rule snippets with a 14-query eval set.
- `agent/clients.py` – live Anthropic client, retry wrapper (exponential backoff, token accounting),
  scripted client for deterministic offline tests.
- `agent/data.py` – synthetic claims with known parameters, policy-limit censoring (~11%), and
  injected defects with exact counts.

## Verified results (`python3 analyze.py`, 100K claims)
- 225/225 injected defects detected.
- EM parameters within 0.8% of truth; a fit ignoring censoring misses the tail mean by 41.8%.
- AIC prefers 2 components over 1 by 23,522; retrieval recall@3 = 14/14.
- Bootstrap 90% CI on the 75K xs 25K layer brackets the true price ($3,652 per claim).

## Run
```bash
python3 -m unittest discover -s tests -v    # 29 tests
python3 analyze.py                              # replay mode: fixed tool plan, no API key
ANTHROPIC_API_KEY=... python3 analyze.py --live # Claude chooses the tools
```
Replay mode executes a fixed tool sequence (used for deterministic runs and CI); only `--live` lets the
model choose tools. Each run writes an audit trail to `audit_log.jsonl`.

## Scope
Built and validated on synthetic data with known ground truth. Not yet validated on real claims.
