"""LossLens end-to-end analysis: validate claims, fit a censored severity model, price an excess layer.

  python3 analyze.py            # replay mode: fixed tool plan, no API key needed
  python3 analyze.py --live     # live mode: Claude chooses the tools (needs ANTHROPIC_API_KEY)
  python3 analyze.py --report   # also write report.html (open it in a browser)
"""
import sys
import time

from agent.core import Agent
from agent.clients import AnthropicClient, RetryingClient, ScriptedClient
from agent.data import build_db, TRUE_PI, TRUE_M1, TRUE_M2
from agent.tools import build_tools, fit_censored_mixed_exp, model_layer_cost
from agent import retrieval
from agent.report import collect_extras, render_report, survival_series

SYSTEM = ("You are an insurance loss-analysis assistant. Check data quality first, retrieve the "
          "relevant rating rules, fit a censored severity curve, compare models, then price the "
          "requested layer with a bootstrap confidence interval. Report defects found, fitted "
          "parameters, model selection, price, interval, and the model-vs-empirical error.")
TASK = "Price a 75,000 xs 25,000 layer on this claims table and flag any data problems first."
A, L = 25000, 75000


def scripted():
    def tu(i, name, args):
        return {"stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "t%d" % i, "name": name, "input": args}]}
    plan = [("data_quality_report", {}), ("search_rating_rules", {"query": "excess layer pricing"}),
            ("fit_severity_curve", {}), ("compare_models", {}),
            ("layer_loss_cost", {"attachment": A, "limit": L}),
            ("bootstrap_layer_ci", {"attachment": A, "limit": L})]
    replies = [tu(i + 1, n, a) for i, (n, a) in enumerate(plan)]
    replies.append({"stop_reason": "end_turn", "content": [{"type": "text", "text": "(scripted run complete)"}]})
    return ScriptedClient(replies)


def main():
    live = "--live" in sys.argv
    t0 = time.time()
    conn, truth = build_db(n_claims=100000)
    tools, lt = build_tools(conn)
    client = RetryingClient(AnthropicClient()) if live else scripted()
    agent = Agent(client, tools, system=SYSTEM, max_steps=10, audit_path="audit_log.jsonl")
    out = agent.run(TASK)
    t1 = time.time()

    fit = lt.fit
    obs = lt._obs()
    naive = fit_censored_mixed_exp([(x, False) for x, _ in obs])  # ignores censoring
    true_cost = model_layer_cost({"pi": TRUE_PI, "mean_1": TRUE_M1, "mean_2": TRUE_M2}, A, L)
    price = lt.layer_loss_cost(A, L)
    ci = lt.bootstrap_layer_ci(A, L)
    sel = lt.compare_models()
    pct = lambda est, tru: 100.0 * abs(est - tru) / tru

    print("rows: %d | injected defects: %d null, %d negative, %d duplicate" % (
        truth["n_rows"], truth["null_losses"], truth["negative_losses"], truth["duplicate_rows"]))
    print("detected:         %s" % lt.data_quality_report())
    print("censored share:   %.1f%% of claims capped at policy limit" % (100 * fit["censored_fraction"]))
    print("EM fit:           pi=%.4f m1=%.1f m2=%.1f  (%d iterations)" % (
        fit["pi"], fit["mean_1"], fit["mean_2"], fit["iterations"]))
    print("param error:      pi %.2f%%  m1 %.2f%%  m2 %.2f%%" % (
        pct(fit["pi"], TRUE_PI), pct(fit["mean_1"], TRUE_M1), pct(fit["mean_2"], TRUE_M2)))
    print("naive (ignores censoring) m2 error: %.1f%%  vs censoring-aware %.2f%%" % (
        pct(naive["mean_2"], TRUE_M2), pct(fit["mean_2"], TRUE_M2)))
    print("model selection:  delta AIC (1-comp minus 2-comp) = %.0f -> %s" % (sel["delta_aic"], sel["preferred"]))
    print("layer %dk xs %dk: model %.1f | true %.1f (err %.2f%%) | empirical %.1f on %d exposed claims (err %.2f%%)" % (
        L // 1000, A // 1000, price["model_cost_per_claim"], true_cost,
        pct(price["model_cost_per_claim"], true_cost), price["empirical_cost_per_claim"],
        price["empirical_n"], abs(price["rel_error_pct"])))
    print("bootstrap %d%% CI: [%.1f, %.1f] (width %.1f%% of point) contains truth: %s" % (
        100 * ci["level"], ci["ci_low"], ci["ci_high"], 100 * (ci["ci_high"] - ci["ci_low"]) / ci["point"],
        ci["ci_low"] <= true_cost <= ci["ci_high"]))
    print("retrieval:        recall@1 %.2f  recall@3 %.2f on %d queries" % (
        retrieval.recall_at_k(1), retrieval.recall_at_k(3), len(retrieval.EVAL_SET)))
    print("agent: %d steps, stopped=%s, tools registered=%d" % (out["steps"], out["stopped"], len(tools)))
    print("total %.1fs" % (t1 - t0))
    if "--report" in sys.argv:
        ctx = {"truth": truth, "quality": lt.data_quality_report(), "fit": fit, "naive": naive,
               "true_params": {"pi": TRUE_PI, "mean_1": TRUE_M1, "mean_2": TRUE_M2}, "sel": sel,
               "price": price, "ci": ci, "true_cost": true_cost, "attachment": A, "limit": L,
               "series": survival_series(obs, fit, naive), "extras": collect_extras(lt, obs),
               "retrieval": {"r1": retrieval.recall_at_k(1), "r3": retrieval.recall_at_k(3),
                             "n": len(retrieval.EVAL_SET)},
               "agent": {"steps": out["steps"], "stopped": out["stopped"], "tools": len(tools)},
               "seconds": t1 - t0}
        with open("report.html", "w") as fh:
            fh.write(render_report(ctx))
        print("report written to report.html")
    if live:
        print("tokens: %s retries: %d\n\n%s" % (client.usage, client.retries, out["text"]))


if __name__ == "__main__":
    main()
