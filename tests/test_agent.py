import json
import os
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.core import Agent
from agent.clients import ScriptedClient, RetryingClient
from agent.data import build_db, TRUE_PI, TRUE_M1, TRUE_M2
from agent.tools import build_tools, fit_censored_mixed_exp, model_layer_cost
from agent import retrieval

CONN, TRUTH = build_db(n_claims=30000, seed=11, n_null=40, n_negative=20, n_dupes=30)
TOOLS, LT = build_tools(CONN, seed=3)


def tool_use(i, name, args):
    return {"stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "t%d" % i, "name": name, "input": args}]}


def final(text):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def true_layer_cost(a, l):
    return model_layer_cost({"pi": TRUE_PI, "mean_1": TRUE_M1, "mean_2": TRUE_M2}, a, l)


class GuardrailTests(unittest.TestCase):
    def test_blocks_drop(self):
        with self.assertRaises(ValueError):
            LT.run_sql("DROP TABLE claims")

    def test_blocks_delete(self):
        with self.assertRaises(ValueError):
            LT.run_sql("DELETE FROM claims")

    def test_multi_statement_rejected(self):
        with self.assertRaises(Exception):
            LT.run_sql("SELECT 1; DELETE FROM claims")

    def test_write_via_cte_blocked_by_readonly_connection(self):
        with self.assertRaises(Exception):
            LT.run_sql("WITH x AS (SELECT 1) INSERT INTO claims VALUES (1,1,2020,'GL',1,1)")

    def test_row_cap(self):
        out = LT.run_sql("SELECT * FROM claims")
        self.assertEqual(len(out["rows"]), 50)
        self.assertTrue(out["truncated"])


class DataQualityTests(unittest.TestCase):
    def test_detects_all_injected_defects(self):
        r = LT.data_quality_report()
        self.assertEqual(r["null_losses"], TRUTH["null_losses"])
        self.assertEqual(r["negative_losses"], TRUTH["negative_losses"])
        self.assertEqual(r["duplicate_rows"], TRUTH["duplicate_rows"])


class CensoredFitTests(unittest.TestCase):
    def test_censoring_is_substantial(self):
        self.assertGreater(TRUTH["censored_fraction"], 0.10)

    def test_em_recovers_parameters_under_censoring(self):
        fit = LT.fit_severity_curve()
        self.assertAlmostEqual(fit["pi"], TRUE_PI, delta=0.04)
        self.assertAlmostEqual(fit["mean_1"] / TRUE_M1, 1.0, delta=0.10)
        self.assertAlmostEqual(fit["mean_2"] / TRUE_M2, 1.0, delta=0.10)

    def test_censoring_aware_fit_beats_naive_fit(self):
        obs = LT._obs()
        aware = fit_censored_mixed_exp(obs)
        naive = fit_censored_mixed_exp([(x, False) for x, _ in obs])
        err = lambda f: abs(f["mean_2"] - TRUE_M2) / TRUE_M2
        self.assertLess(err(aware), err(naive))

    def test_loglik_nondecreasing_improves_with_more_iterations(self):
        obs = LT._obs()[:5000]
        short = fit_censored_mixed_exp(obs, max_iter=3)
        long_ = fit_censored_mixed_exp(obs, max_iter=200)
        self.assertGreater(long_["loglik"], short["loglik"])


class ModelSelectionAndPricingTests(unittest.TestCase):
    def test_two_component_preferred_by_aic_and_bic(self):
        LT.fit_severity_curve()
        r = LT.compare_models()
        self.assertEqual(r["preferred"], "two_component")
        self.assertGreater(r["delta_aic"], 100)

    def test_layer_cost_close_to_empirical_on_exposed_claims(self):
        LT.fit_severity_curve()
        r = LT.layer_loss_cost(25000, 25000)
        self.assertLess(abs(r["rel_error_pct"]), 5.0)

    def test_layer_cost_close_to_true_value(self):
        LT.fit_severity_curve()
        r = LT.layer_loss_cost(25000, 75000)
        truth = true_layer_cost(25000, 75000)
        self.assertLess(abs(r["model_cost_per_claim"] - truth) / truth, 0.03)

    def test_layer_requires_fit(self):
        _, fresh = build_tools(CONN)
        with self.assertRaises(ValueError):
            fresh.layer_loss_cost(1000, 1000)

    def test_closed_form_layer_cost(self):
        import math
        fit = {"pi": 1.0, "mean_1": 1000.0, "mean_2": 5000.0}
        self.assertAlmostEqual(model_layer_cost(fit, 1000.0, 1000.0),
                               1000.0 * (math.exp(-1.0) - math.exp(-2.0)), places=6)

    def test_bootstrap_ci_brackets_point_and_truth(self):
        LT.fit_severity_curve()
        ci = LT.bootstrap_layer_ci(25000, 75000, n_boot=20, sample_size=6000)
        self.assertLess(ci["ci_low"], ci["point"])
        self.assertGreater(ci["ci_high"], ci["point"])
        truth = true_layer_cost(25000, 75000)
        self.assertLessEqual(ci["ci_low"], truth)
        self.assertGreaterEqual(ci["ci_high"], truth)


class RetrievalTests(unittest.TestCase):
    def test_recall_at_3(self):
        self.assertGreaterEqual(retrieval.recall_at_k(3), 0.9)

    def test_top_hit_for_limit_censoring(self):
        top = retrieval.INDEX.search("what happens to losses above the policy limit", 1)[0]
        self.assertEqual(top["id"], "R02")

    def test_tool_returns_k_results(self):
        self.assertEqual(len(LT.search_rating_rules("excess layer", 2)), 2)


class ClientResilienceTests(unittest.TestCase):
    class Flaky(object):
        def __init__(self, fails, code=429):
            self.fails, self.code, self.calls = fails, code, 0

        def create(self, system, messages, tools):
            self.calls += 1
            if self.calls <= self.fails:
                raise urllib.error.HTTPError("u", self.code, "err", {}, None)
            return {"stop_reason": "end_turn", "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 5}}

    def test_exponential_backoff_then_success(self):
        delays = []
        c = RetryingClient(self.Flaky(2), base_delay=1.0, sleep=delays.append)
        c.create("", [], [])
        self.assertEqual(delays, [1.0, 2.0])
        self.assertEqual(c.retries, 2)

    def test_non_retryable_error_raises_immediately(self):
        delays = []
        c = RetryingClient(self.Flaky(1, code=400), sleep=delays.append)
        with self.assertRaises(urllib.error.HTTPError):
            c.create("", [], [])
        self.assertEqual(delays, [])

    def test_gives_up_after_max_retries(self):
        c = RetryingClient(self.Flaky(99), max_retries=2, sleep=lambda s: None)
        with self.assertRaises(urllib.error.HTTPError):
            c.create("", [], [])

    def test_token_usage_accumulates(self):
        c = RetryingClient(self.Flaky(0), sleep=lambda s: None)
        c.create("", [], [])
        c.create("", [], [])
        self.assertEqual(c.usage, {"input_tokens": 20, "output_tokens": 10})


class AgentLoopTests(unittest.TestCase):
    def _agent(self, replies, **kw):
        return Agent(ScriptedClient(replies), TOOLS, system="test", **kw)

    def test_runs_tools_then_finishes(self):
        a = self._agent([tool_use(1, "data_quality_report", {}), final("done")])
        out = a.run("check quality")
        self.assertEqual((out["stopped"], out["steps"]), ("end_turn", 2))
        self.assertTrue(a.trace[0]["ok"])

    def test_tool_error_is_returned_not_raised(self):
        a = self._agent([tool_use(1, "run_sql", {"query": "DROP TABLE claims"}), final("ok")])
        out = a.run("try to drop")
        self.assertTrue(out["messages"][2]["content"][0]["is_error"])
        self.assertFalse(a.trace[0]["ok"])

    def test_unknown_tool(self):
        out = self._agent([tool_use(1, "nope", {}), final("ok")]).run("x")
        self.assertTrue(out["messages"][2]["content"][0]["is_error"])

    def test_max_steps_budget(self):
        a = self._agent([tool_use(i, "data_quality_report", {}) for i in range(1, 10)], max_steps=3)
        out = a.run("loop forever")
        self.assertEqual((out["stopped"], out["steps"]), ("max_steps", 3))

    def test_audit_log_written(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            self._agent([tool_use(1, "data_quality_report", {}), final("done")], audit_path=path).run("go")
            with open(path) as fh:
                lines = [json.loads(l) for l in fh]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["tool"], "data_quality_report")

    def test_full_pricing_workflow(self):
        a = self._agent([tool_use(1, "data_quality_report", {}),
                         tool_use(2, "search_rating_rules", {"query": "excess layer pricing"}),
                         tool_use(3, "fit_severity_curve", {}),
                         tool_use(4, "compare_models", {}),
                         tool_use(5, "layer_loss_cost", {"attachment": 25000, "limit": 75000}),
                         tool_use(6, "bootstrap_layer_ci", {"attachment": 25000, "limit": 75000,
                                                            "n_boot": 5, "sample_size": 2000}),
                         final("priced")])
        out = a.run("price the 75k xs 25k layer")
        self.assertEqual(out["steps"], 7)
        self.assertTrue(all(t.get("ok", True) for t in a.trace))


if __name__ == "__main__":
    unittest.main()
