"""Synthetic claims data (no real or employer data).

Ground-up severity is a 2-component mixed exponential with known parameters. Each policy has a
limit, and paid loss is capped at that limit (right-censored), as in real claims data. Known-count
data-quality defects are injected so tools can be tested against ground truth."""
import random
import sqlite3

TRUE_PI, TRUE_M1, TRUE_M2 = 0.8, 5000.0, 40000.0
LINES = ["Property", "GL", "Auto", "WC"]
LIMITS = [10000, 25000, 50000, 100000]


def build_db(n_claims=100000, seed=7, n_null=100, n_negative=50, n_dupes=75):
    """Returns (connection, truth). Defects (exact counts): n_null NULL losses, n_negative negative
    losses, n_dupes duplicated claim_id rows."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE claims (claim_id INTEGER, policy_id INTEGER, accident_year INTEGER,"
                 " line TEXT, policy_limit REAL, loss_amount REAL)")
    rows, n_cens = [], 0
    for i in range(1, n_claims + 1):
        mean = TRUE_M1 if rng.random() < TRUE_PI else TRUE_M2
        gross = rng.expovariate(1.0 / mean)
        limit = rng.choice(LIMITS)
        paid = min(gross, limit)
        n_cens += gross >= limit
        rows.append([i, rng.randint(1, n_claims // 3), rng.randint(2015, 2024),
                     rng.choice(LINES), float(limit), round(paid, 2)])
    idx = rng.sample(range(n_claims), n_null + n_negative)
    for k in idx[:n_null]:
        rows[k][5] = None
    for k in idx[n_null:]:
        rows[k][5] = -abs(rows[k][5])
    taken = set(idx)
    clean_idx = [k for k in range(n_claims) if k not in taken]
    for k in rng.sample(clean_idx, n_dupes):
        rows.append(list(rows[k]))
    conn.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.execute("PRAGMA query_only = ON")  # agent SQL cannot modify the data
    truth = {"n_rows": len(rows), "null_losses": n_null, "negative_losses": n_negative,
             "duplicate_rows": n_dupes, "pi": TRUE_PI, "m1": TRUE_M1, "m2": TRUE_M2,
             "censored_fraction": n_cens / float(n_claims)}
    return conn, truth
