"""Domain tools the agent can call.

Statistics: right-censored 2-component mixed exponential via EM (paid losses are capped at policy
limits), AIC/BIC model selection against a 1-component fit, layer pricing with an empirical check on
fully-exposed claims, and bootstrap confidence intervals for layer prices."""
import math
import random
import re

from .core import Tool
from .retrieval import INDEX

MAX_ROWS = 50
_SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def fit_censored_mixed_exp(obs, init=None, tol=1e-9, max_iter=1000):
    """EM for f(x) = pi/m1 e^(-x/m1) + (1-pi)/m2 e^(-x/m2) with right-censoring.

    obs: list of (x, censored) where a censored x is a lower bound (paid at the policy limit).
    E-step for censored points uses the survival function for the posterior and the memoryless
    property, E[X | X > c, component i] = c + m_i."""
    unc = [x for x, c in obs if not c]
    cen = [x for x, c in obs if c]
    n = len(obs)
    mean = sum(x for x, _ in obs) / float(n)
    pi, m1, m2 = init if init else (0.5, mean * 0.5, mean * 2.0)
    prev = None
    for it in range(1, max_iter + 1):
        sr = s1 = s2 = ll = 0.0
        for x in unc:
            a = pi * math.exp(-x / m1) / m1
            b = (1.0 - pi) * math.exp(-x / m2) / m2
            t = a + b
            r = a / t
            ll += math.log(t)
            sr += r
            s1 += r * x
            s2 += (1.0 - r) * x
        for c in cen:
            a = pi * math.exp(-c / m1)
            b = (1.0 - pi) * math.exp(-c / m2)
            t = a + b
            r = a / t
            ll += math.log(t)
            sr += r
            s1 += r * (c + m1)
            s2 += (1.0 - r) * (c + m2)
        pi, m1, m2 = sr / n, s1 / sr, s2 / (n - sr)
        if prev is not None and abs(ll - prev) <= tol * abs(ll):
            break
        prev = ll
    if m1 > m2:
        pi, m1, m2 = 1.0 - pi, m2, m1
    return {"pi": pi, "mean_1": m1, "mean_2": m2, "loglik": ll, "iterations": it,
            "n_claims_used": n, "censored_fraction": len(cen) / float(n)}


def fit_censored_exp(obs):
    """Closed-form censored exponential MLE: mean = sum(x) / (# uncensored)."""
    n_unc = sum(1 for _, c in obs if not c)
    total = sum(x for x, _ in obs)
    m = total / n_unc
    return {"mean": m, "loglik": -n_unc * math.log(m) - total / m}


def _lev(m, d):
    return m * (1.0 - math.exp(-d / m))


def model_layer_cost(fit, attachment, limit):
    top = attachment + limit
    return (fit["pi"] * (_lev(fit["mean_1"], top) - _lev(fit["mean_1"], attachment))
            + (1.0 - fit["pi"]) * (_lev(fit["mean_2"], top) - _lev(fit["mean_2"], attachment)))


class LossTools(object):
    def __init__(self, conn, seed=0):
        self.conn = conn
        self.fit = None
        self.rng = random.Random(seed)

    def _clean_rows(self):
        cur = self.conn.execute(
            "SELECT MIN(loss_amount), MIN(policy_limit) FROM claims WHERE loss_amount > 0 "
            "GROUP BY claim_id")
        return [(r[0], r[1]) for r in cur]

    def _obs(self):
        return [(x, x >= lim) for x, lim in self._clean_rows()]

    def run_sql(self, query):
        if not _SELECT_ONLY.match(query):
            raise ValueError("only SELECT / WITH queries are allowed")
        cur = self.conn.execute(query.strip().rstrip(";"))  # sqlite3 rejects multi-statement input
        rows = cur.fetchmany(MAX_ROWS + 1)
        cols = [d[0] for d in cur.description]
        return {"columns": cols, "rows": rows[:MAX_ROWS], "truncated": len(rows) > MAX_ROWS}

    def data_quality_report(self):
        q = lambda s: self.conn.execute(s).fetchone()[0]
        return {
            "rows": q("SELECT COUNT(*) FROM claims"),
            "null_losses": q("SELECT COUNT(*) FROM claims WHERE loss_amount IS NULL"),
            "negative_losses": q("SELECT COUNT(*) FROM claims WHERE loss_amount < 0"),
            "duplicate_rows": q("SELECT COUNT(*) - COUNT(DISTINCT claim_id) FROM claims"),
        }

    def fit_severity_curve(self):
        self.fit = fit_censored_mixed_exp(self._obs())
        return self.fit

    def compare_models(self):
        """AIC/BIC of 1- vs 2-component censored exponential fits on the same data."""
        obs = self._obs()
        two = self.fit if self.fit else fit_censored_mixed_exp(obs)
        one = fit_censored_exp(obs)
        n = len(obs)
        ic = lambda ll, k: {"aic": 2 * k - 2 * ll, "bic": k * math.log(n) - 2 * ll}
        a1, a2 = ic(one["loglik"], 1), ic(two["loglik"], 3)
        return {"one_component": a1, "two_component": a2,
                "delta_aic": a1["aic"] - a2["aic"], "preferred": "two_component"
                if a2["aic"] < a1["aic"] and a2["bic"] < a1["bic"] else "one_component"}

    def layer_loss_cost(self, attachment, limit):
        """Model price per claim vs an empirical check restricted to claims whose policy limit is
        at least the layer top, since only those claims reveal the whole layer (uncensored)."""
        if self.fit is None:
            raise ValueError("call fit_severity_curve first")
        top = attachment + limit
        rows = [x for x, lim in self._clean_rows() if lim >= top]
        if not rows:
            raise ValueError("no claims have a policy limit covering the layer")
        emp = sum(min(max(x - attachment, 0.0), limit) for x in rows) / len(rows)
        mod = model_layer_cost(self.fit, attachment, limit)
        return {"attachment": attachment, "limit": limit, "model_cost_per_claim": mod,
                "empirical_cost_per_claim": emp, "empirical_n": len(rows),
                "rel_error_pct": 100.0 * (mod - emp) / emp}

    def bootstrap_layer_ci(self, attachment, limit, n_boot=30, sample_size=10000, level=0.90):
        """Nonparametric bootstrap of the layer price: resample claims, refit (warm-started from the
        full-data fit), reprice. Returns the percentile interval."""
        if self.fit is None:
            raise ValueError("call fit_severity_curve first")
        obs = self._obs()
        init = (self.fit["pi"], self.fit["mean_1"], self.fit["mean_2"])
        prices = []
        for _ in range(n_boot):
            samp = [obs[self.rng.randrange(len(obs))] for _ in range(sample_size)]
            f = fit_censored_mixed_exp(samp, init=init)
            prices.append(model_layer_cost(f, attachment, limit))
        prices.sort()
        lo = prices[int((1 - level) / 2 * n_boot)]
        hi = prices[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
        return {"point": model_layer_cost(self.fit, attachment, limit), "ci_low": lo, "ci_high": hi,
                "level": level, "n_boot": n_boot, "sample_size": sample_size, "prices": prices}

    def search_rating_rules(self, query, k=3):
        return INDEX.search(query, k)


def build_tools(conn, seed=0):
    lt = LossTools(conn, seed=seed)
    obj = lambda props, req=None: {"type": "object", "properties": props, "required": req or []}
    num = {"type": "number"}
    tools = [
        Tool("run_sql", "Run a read-only SELECT against claims(claim_id, policy_id, accident_year, "
             "line, policy_limit, loss_amount). Returns at most %d rows." % MAX_ROWS,
             obj({"query": {"type": "string"}}, ["query"]), lt.run_sql),
        Tool("data_quality_report", "Count null losses, negative losses, and duplicate claim ids.",
             obj({}), lt.data_quality_report),
        Tool("fit_severity_curve", "Fit a right-censored 2-component mixed exponential severity "
             "curve (paid losses capped at policy limits) to cleaned claims using EM.",
             obj({}), lt.fit_severity_curve),
        Tool("compare_models", "Compare 1- vs 2-component censored exponential fits by AIC and BIC.",
             obj({}), lt.compare_models),
        Tool("layer_loss_cost", "Expected loss per claim in a layer (limit xs attachment) from the "
             "fitted curve versus an empirical check. Requires fit_severity_curve first.",
             obj({"attachment": num, "limit": num}, ["attachment", "limit"]), lt.layer_loss_cost),
        Tool("bootstrap_layer_ci", "Bootstrap confidence interval for a layer price. Requires "
             "fit_severity_curve first.", obj({"attachment": num, "limit": num},
                                              ["attachment", "limit"]), lt.bootstrap_layer_ci),
        Tool("search_rating_rules", "Retrieve the most relevant rating-rule snippets (BM25) for a "
             "question about limits, deductibles, layers, trending, or credibility.",
             obj({"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
             lt.search_rating_rules),
    ]
    return tools, lt
