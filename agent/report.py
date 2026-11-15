"""Self-contained HTML report (inline SVG, no external assets) for a LossLens analysis run."""
import bisect
import html
import json
import math

BASE = ("(SELECT claim_id, MIN(line) line, MIN(accident_year) yr, MIN(policy_limit) lim, "
        "MIN(loss_amount) amt FROM claims WHERE loss_amount > 0 GROUP BY claim_id)")


# ---------------------------------------------------------------- data prep
def km_curve(obs):
    """Kaplan-Meier survival steps from (x, censored) pairs."""
    data = sorted(obs, key=lambda o: (o[0], o[1]))
    at_risk, s, xs, ss, i = len(data), 1.0, [], [], 0
    while i < len(data):
        x, deaths, j = data[i][0], 0, i
        while j < len(data) and data[j][0] == x:
            deaths += 0 if data[j][1] else 1
            j += 1
        if deaths:
            s *= 1.0 - deaths / float(at_risk)
            xs.append(x)
            ss.append(s)
        at_risk -= j - i
        i = j
    return xs, ss


def _surv_mix(f, x):
    return f["pi"] * math.exp(-x / f["mean_1"]) + (1 - f["pi"]) * math.exp(-x / f["mean_2"])


def survival_series(obs, fit, naive, x_max=100000, points=100):
    kx, ks = km_curve(obs)
    grid = [x_max * i / float(points) for i in range(points + 1)]

    def km(x):
        k = bisect.bisect_right(kx, x)
        return ks[k - 1] if k else 1.0
    return {"x": grid, "empirical": [km(x) for x in grid],
            "fitted": [_surv_mix(fit, x) for x in grid], "naive": [_surv_mix(naive, x) for x in grid]}


def collect_extras(lt, obs, bin_width=2500, x_max=100000):
    """Descriptive aggregates on the cleaned claims table for the report's secondary charts."""
    q = lt.conn.execute
    nb = x_max // bin_width
    counts = [0] * nb
    for x, _ in obs:
        counts[min(int(x // bin_width), nb - 1)] += 1
    return {
        "hist": {"counts": counts, "width": bin_width},
        "by_limit": [(r[0], r[1], r[2]) for r in q(
            "SELECT lim, COUNT(*), SUM(amt >= lim) FROM %s GROUP BY lim ORDER BY lim" % BASE)],
        "by_line": [(r[0], r[1], r[2]) for r in q(
            "SELECT line, COUNT(*), AVG(amt) FROM %s GROUP BY line ORDER BY AVG(amt) DESC" % BASE)],
        "by_year": [(r[0], r[1]) for r in q(
            "SELECT yr, COUNT(*) FROM %s GROUP BY yr ORDER BY yr" % BASE)],
    }


# ---------------------------------------------------------------- formatting
def _money(v):
    return "${:,.0f}".format(v)


def _short(v):
    v = float(v)
    if abs(v) >= 1e6:
        return "%.1fM" % (v / 1e6)
    if abs(v) >= 1e3:
        return "%gk" % round(v / 1e3, 1)
    return "%g" % round(v, 1)


def _tip(s):
    return html.escape(s, quote=True)


def _nice_max(v):
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= v:
            return m * e
    return 10 * e


def _rtop(x, y, w, h, r=4):
    r = max(0.0, min(r, w / 2.0, h))
    return ("M%.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Z" % (
        x, y + h, y + r, x, y, x + r, y, x + w - r, x + w, y, x + w, y + r, y + h))


def _svg(w, h, label, inner, extra=""):
    return '<svg viewBox="0 0 %d %d" role="img" aria-label="%s"%s>%s</svg>' % (
        w, h, _tip(label), extra, inner)


# ---------------------------------------------------------------- charts
def survival_chart(ser, y_min=0.01):
    W, H, ML, MR, MT, MB = 760, 340, 52, 20, 12, 40
    xmax = ser["x"][-1]
    px = lambda x: ML + (W - ML - MR) * x / xmax
    py = lambda y: MT + (H - MT - MB) * (math.log10(max(y, y_min)) / math.log10(y_min))
    g = []
    for t in (0.01, 0.1, 1.0):
        g.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>'
                 '<text class="tick" x="%d" y="%.1f" text-anchor="end">%g%%</text>' % (
                     ML, W - MR, py(t), py(t), ML - 10, py(t) + 4, t * 100))
    for x in range(0, int(xmax) + 1, 25000):
        g.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">$%dk</text>' % (
            px(x), H - MB + 22, x // 1000))
    path = lambda ys: " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", px(x), py(y))
                               for i, (x, y) in enumerate(zip(ser["x"], ys)))
    g.append('<path class="line l-naive" d="%s"/>' % path(ser["naive"]))
    g.append('<path class="line l-emp" d="%s"/>' % path(ser["empirical"]))
    g.append('<path class="line l-fit" d="%s"/>' % path(ser["fitted"]))
    g.append('<line id="xh" class="xh" y1="%d" y2="%d" visibility="hidden"/>'
             '<rect id="hit" x="%d" y="%d" width="%d" height="%d" fill="transparent"/>' % (
                 MT, H - MB, ML, MT, W - ML - MR, H - MT - MB))
    return _svg(W, H, "Survival curves: empirical, censoring-aware fit, naive fit", "".join(g),
                ' id="surv"')


def layer_chart(price, ci, true_cost):
    vals = [ci["ci_low"], ci["ci_high"], true_cost, price["model_cost_per_claim"],
            price["empirical_cost_per_claim"]]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.2
    lo, hi = lo - pad, hi + pad
    w, h, ml, mr = 760, 170, 20, 20
    px = lambda v: ml + (w - ml - mr) * (v - lo) / (hi - lo)
    step = 10 ** math.floor(math.log10((hi - lo) / 4))
    step *= 5 if (hi - lo) / step > 20 else (2 if (hi - lo) / step > 8 else 1)
    g, t = [], math.ceil(lo / step) * step
    while t <= hi:
        g.append('<line class="grid" x1="%.1f" x2="%.1f" y1="34" y2="112"/>'
                 '<text class="tick" x="%.1f" y="136" text-anchor="middle">%s</text>' % (
                     px(t), px(t), px(t), _money(t)))
        t += step
    g.append('<rect class="ciband" x="%.1f" y="56" width="%.1f" height="30" rx="15" data-tip="%s"/>' % (
        px(ci["ci_low"]), px(ci["ci_high"]) - px(ci["ci_low"]),
        _tip("<b>%d%% bootstrap interval</b><br>%s to %s" % (
            100 * ci["level"], _money(ci["ci_low"]), _money(ci["ci_high"])))))
    g.append('<text class="lbl" x="%.1f" y="48" text-anchor="middle">%d%% interval</text>' % (
        (px(ci["ci_low"]) + px(ci["ci_high"])) / 2, 100 * ci["level"]))
    g.append('<line class="truth" x1="%.1f" x2="%.1f" y1="34" y2="112"/>'
             '<text class="lbl" x="%.1f" y="26" text-anchor="middle">True %s</text>' % (
                 px(true_cost), px(true_cost), px(true_cost), _money(true_cost)))
    g.append('<circle class="pt-model" cx="%.1f" cy="71" r="8" data-tip="%s"/>' % (
        px(price["model_cost_per_claim"]),
        _tip("<b>Model price</b><br>" + _money(price["model_cost_per_claim"]))))
    g.append('<circle class="pt-emp" cx="%.1f" cy="71" r="6" data-tip="%s"/>' % (
        px(price["empirical_cost_per_claim"]),
        _tip("<b>Empirical check</b><br>" + _money(price["empirical_cost_per_claim"]))))
    return _svg(w, h, "Layer price with bootstrap interval", "".join(g))


def hist_chart(counts, edges, cls="b0", w=370, h=230, xfmt=_short, markers=(), tip_fmt=None,
               y_label_fmt=_short):
    """Contiguous histogram. counts[i] spans edges[i]..edges[i+1]; markers = [(x, label)]."""
    ml, mr, mt, mb = 40, 8, 22, 30
    ymax = _nice_max(max(counts))
    x0, x1 = edges[0], edges[-1]
    px = lambda x: ml + (w - ml - mr) * (x - x0) / float(x1 - x0)
    py = lambda y: mt + (h - mt - mb) * (1 - y / ymax)
    g = []
    for k in range(5):
        v = ymax * k / 4.0
        g.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>'
                 '<text class="tick" x="%d" y="%.1f" text-anchor="end">%s</text>' % (
                     ml, w - mr, py(v), py(v), ml - 6, py(v) + 4, y_label_fmt(v)))
    for i, c in enumerate(counts):
        bx, bw = px(edges[i]) + 0.75, px(edges[i + 1]) - px(edges[i]) - 1.5
        tip = tip_fmt(i, c) if tip_fmt else "%s&ndash;%s<br><b>%s</b>" % (
            xfmt(edges[i]), xfmt(edges[i + 1]), "{:,}".format(c) if isinstance(c, int) else c)
        g.append('<path class="%s" d="%s" data-tip="%s"/>' % (
            cls, _rtop(bx, py(c), bw, py(0) - py(c), 2.5), _tip(tip)))
    n_ticks = 5
    for k in range(n_ticks + 1):
        x = x0 + (x1 - x0) * k / float(n_ticks)
        g.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (
            px(x), h - 8, xfmt(x)))
    for x, label in markers:
        g.append('<line class="mk" x1="%.1f" x2="%.1f" y1="%d" y2="%.1f"/>'
                 '<text class="lbl" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (
                     px(x), px(x), mt - 6, py(0), px(x), mt - 10, label))
    return _svg(w, h, "Histogram", "".join(g))


def vbar_chart(cats, series, fmt=_short, w=370, h=230, tip_fmt=None):
    """Grouped vertical bars. series = [(name, values, css_class)]."""
    ml, mr, mt, mb = 40, 8, 22, 30
    ymax = _nice_max(max(max(v) for _, v, _ in series))
    gw = (w - ml - mr) / float(len(cats))
    bw = min(34.0, gw * 0.72 / len(series))
    py = lambda y: mt + (h - mt - mb) * (1 - y / ymax)
    g = []
    for k in range(5):
        v = ymax * k / 4.0
        g.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>'
                 '<text class="tick" x="%d" y="%.1f" text-anchor="end">%s</text>' % (
                     ml, w - mr, py(v), py(v), ml - 6, py(v) + 4, fmt(v)))
    label_values = len(cats) * len(series) <= 12
    for i, c in enumerate(cats):
        total = bw * len(series) + 3 * (len(series) - 1)
        x = ml + gw * i + (gw - total) / 2.0
        for j, (name, vals, cls) in enumerate(series):
            bx = x + j * (bw + 3)
            tip = tip_fmt(c, name, vals[i]) if tip_fmt else "<b>%s</b><br>%s: %s" % (c, name, fmt(vals[i]))
            g.append('<path class="%s" d="%s" data-tip="%s"/>' % (
                cls, _rtop(bx, py(vals[i]), bw, py(0) - py(vals[i]), 5), _tip(tip)))
            if label_values:
                g.append('<text class="val" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (
                    bx + bw / 2, py(vals[i]) - 6, fmt(vals[i])))
        g.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (
            ml + gw * (i + 0.5), h - 8, c))
    return _svg(w, h, "Bar chart", "".join(g))


def hbar_chart(rows, fmt, w=370, label_w=118, bar_h=11, tip_fmt=None):
    """rows = [(label, [(value, css_class, name)])]. Bars share one scale."""
    vmax = _nice_max(max(v for _, bars in rows for v, _, _ in bars))
    row_h = [len(bars) * (bar_h + 3) + 16 for _, bars in rows]
    h = int(sum(row_h)) + 4
    span = w - label_w - 56
    g, y = [], 2
    for (label, bars), rh in zip(rows, row_h):
        g.append('<text class="tick2" x="0" y="%.1f">%s</text>' % (y + rh / 2.0 + 1, label))
        by = y + 6
        for v, cls, name in bars:
            bw = max(span * v / vmax, 1.5)
            tip = tip_fmt(label, name, v) if tip_fmt else "<b>%s</b><br>%s: %s" % (label, name, fmt(v))
            g.append('<rect class="%s" x="%d" y="%.1f" width="%.1f" height="%d" rx="%.1f" data-tip="%s"/>'
                     '<text class="val" x="%.1f" y="%.1f">%s</text>' % (
                         cls, label_w, by, bw, bar_h, bar_h / 2.0, _tip(tip),
                         label_w + bw + 8, by + bar_h - 2, fmt(v)))
            by += bar_h + 3
        y += rh
    return _svg(w, h, "Bar chart", "".join(g))


# ---------------------------------------------------------------- page
CSS = """
:root{color-scheme:light;--bg:#f5f5f7;--card:#fff;--ink:#1d1d1f;--ink2:#6e6e73;--ink3:#86868b;
--line:#e8e8ed;--blue:#0071e3;--orange:#ff9500;--gray:#c7c7cc;--band:rgba(0,113,227,.14);--nav:rgba(245,245,247,.8)}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#000;--card:#1c1c1e;--ink:#f5f5f7;
--ink2:#a1a1a6;--ink3:#86868b;--line:#2c2c2e;--blue:#2997ff;--orange:#ff9f0a;--gray:#48484a;
--band:rgba(41,151,255,.2);--nav:rgba(0,0,0,.72)}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased;
font:17px/1.47 -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Helvetica,Arial,sans-serif}
nav{position:sticky;top:0;z-index:5;backdrop-filter:saturate(180%) blur(20px);-webkit-backdrop-filter:saturate(180%) blur(20px);
background:var(--nav);border-bottom:1px solid var(--line)}
nav div{max-width:1040px;margin:0 auto;padding:0 22px;height:48px;display:flex;align-items:center;gap:26px;font-size:13px}
nav b{font-weight:600;font-size:15px;margin-right:auto;letter-spacing:-.01em}
nav a{color:var(--ink2);text-decoration:none}nav a:hover{color:var(--ink)}
main{max-width:1040px;margin:0 auto;padding:0 22px 96px}
.hero{padding:88px 0 56px;text-align:center}
.eyebrow{font-size:17px;font-weight:600;color:var(--orange);margin:0 0 8px}
h1{font-size:clamp(36px,6vw,64px);line-height:1.07;letter-spacing:-.03em;font-weight:700;margin:0 auto;max-width:820px}
.lede{font-size:clamp(18px,2.2vw,24px);line-height:1.3;color:var(--ink2);max-width:660px;margin:20px auto 0;letter-spacing:-.01em}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin:56px 0 0;text-align:center}
.stat{padding:8px 12px}.stat+.stat{border-left:1px solid var(--line)}
.stat b{display:block;font-size:clamp(32px,4.4vw,52px);font-weight:700;letter-spacing:-.03em;line-height:1.1;font-variant-numeric:tabular-nums}
.stat span{display:block;margin-top:6px;font-size:14px;color:var(--ink2)}
section.block{padding-top:72px}
h2{font-size:clamp(28px,3.6vw,40px);letter-spacing:-.025em;line-height:1.1;font-weight:700;margin:0}
.kicker{color:var(--ink2);font-size:19px;line-height:1.4;max-width:640px;margin:10px 0 26px;letter-spacing:-.01em}
.card{background:var(--card);border-radius:22px;padding:28px 28px 22px;overflow:hidden}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}
.card h3{font-size:19px;font-weight:600;margin:0;letter-spacing:-.015em}
.card p.s{font-size:14px;color:var(--ink2);margin:2px 0 14px;line-height:1.4}
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}.tick{fill:var(--ink3);font-size:11.5px}.tick2{fill:var(--ink2);font-size:13px;dominant-baseline:middle}
.val{fill:var(--ink2);font-size:11.5px;font-variant-numeric:tabular-nums}.lbl{fill:var(--ink);font-size:12px;font-weight:500}
.line{fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.l-fit{stroke:var(--blue)}
.l-naive{stroke:var(--orange);stroke-dasharray:1 6}.l-emp{stroke:var(--ink3);stroke-width:1.5}
.xh{stroke:var(--ink3);stroke-width:1}.mk{stroke:var(--ink3);stroke-width:1;stroke-dasharray:2 3}
.ciband{fill:var(--band)}.truth{stroke:var(--ink);stroke-width:2;stroke-linecap:round}
.pt-model{fill:var(--blue);stroke:var(--card);stroke-width:3}.pt-emp{fill:var(--orange);stroke:var(--card);stroke-width:2}
.b0{fill:var(--blue)}.b1{fill:var(--gray)}.b2{fill:var(--orange)}
.b0:hover,.b1:hover,.b2:hover,.ciband:hover{opacity:.78}
rect.b0,rect.b1,rect.b2{}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:13px;color:var(--ink2);margin-top:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:-1px}
.legend i.ln{width:18px;height:0;border-radius:0;border-top:2.5px solid;vertical-align:3px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px 28px;margin:0}
.facts div{border-top:1px solid var(--line);padding-top:12px}
.facts dt{font-size:13px;color:var(--ink2)}.facts dd{margin:2px 0 0;font-size:19px;font-weight:600;letter-spacing:-.015em}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:15px}
th,td{text-align:right;padding:11px 4px;border-bottom:1px solid var(--line)}tr:last-child td{border-bottom:0}
th:first-child,td:first-child{text-align:left}th{font-weight:500;color:var(--ink2);font-size:13px}
.foot{margin:72px 0 0;text-align:center;color:var(--ink3);font-size:13px}
.tip{position:fixed;z-index:10;pointer-events:none;display:none;background:rgba(30,30,32,.92);color:#f5f5f7;
backdrop-filter:blur(12px);border-radius:10px;padding:8px 12px;font-size:12.5px;line-height:1.45;
box-shadow:0 8px 28px rgba(0,0,0,.25)}
@media(max-width:760px){.stats{grid-template-columns:1fr 1fr;row-gap:28px}.stat:nth-child(3){border-left:0}
.grid2{grid-template-columns:1fr}.hero{padding-top:56px}nav a{display:none}.card{padding:22px 18px 16px}}
"""

JS = """
const D=%s,svg=document.getElementById('surv'),hit=document.getElementById('hit'),
xh=document.getElementById('xh'),tip=document.getElementById('tip');
const ML=52,MR=20,W=760;
function place(e){tip.style.display='block';const r=tip.getBoundingClientRect();
tip.style.left=Math.min(e.clientX+16,innerWidth-r.width-8)+'px';tip.style.top=(e.clientY+16)+'px'}
hit.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect(),sx=(e.clientX-r.left)*W/r.width;
const f=Math.max(0,Math.min(1,(sx-ML)/(W-ML-MR))),i=Math.round(f*(D.x.length-1)),
X=ML+(W-ML-MR)*i/(D.x.length-1),p=v=>(100*v).toFixed(2)+'%%';
xh.setAttribute('x1',X);xh.setAttribute('x2',X);xh.setAttribute('visibility','visible');
tip.innerHTML='<b>Claims above $'+Math.round(D.x[i]/1000)+'k</b><br>Empirical '+p(D.empirical[i])+
'<br>Censoring-aware '+p(D.fitted[i])+'<br>Naive '+p(D.naive[i]);place(e)});
hit.addEventListener('mouseleave',()=>{tip.style.display='none';xh.setAttribute('visibility','hidden')});
document.addEventListener('mousemove',e=>{const t=e.target.closest&&e.target.closest('[data-tip]');
if(!t){if(e.target!==hit)tip.style.display='none';return}tip.innerHTML=t.getAttribute('data-tip');place(e)});
"""


def _card(title, sub, body, legend=""):
    return '<div class="card"><h3>%s</h3><p class="s">%s</p>%s%s</div>' % (title, sub, body, legend)


def _legend(items):
    return '<div class="legend">%s</div>' % "".join(
        '<span><i class="%s" style="%s"></i>%s</span>' % (c, s, t) for t, c, s in items)


def render_report(ctx):
    f, tr, nv = ctx["fit"], ctx["true_params"], ctx["naive"]
    pct = lambda est, tru: 100.0 * abs(est - tru) / tru
    price, ci, q, truth, ex = ctx["price"], ctx["ci"], ctx["quality"], ctx["truth"], ctx["extras"]
    sel, ret, ag = ctx["sel"], ctx["retrieval"], ctx["agent"]
    keys = ("null_losses", "negative_losses", "duplicate_rows")
    detected, injected = sum(q[k] for k in keys), sum(truth[k] for k in keys)
    err = pct(price["model_cost_per_claim"], ctx["true_cost"])
    covers = ci["ci_low"] <= ctx["true_cost"] <= ci["ci_high"]
    A, L = ctx["attachment"], ctx["limit"]
    layer = "$%dk xs $%dk" % (L // 1000, A // 1000)

    stats = [(_money(price["model_cost_per_claim"]), "model price per claim"),
             ("%.1f%%" % err, "error vs true price"),
             ("%d/%d" % (detected, injected), "data defects caught"),
             ("%.0f%%" % (100 * f["censored_fraction"]), "claims capped at limit")]
    stats_html = "".join('<div class="stat"><b>%s</b><span>%s</span></div>' % s for s in stats)

    surv = _card("Probability a claim exceeds a given size",
                 "Log scale. The censoring-aware fit follows the Kaplan&ndash;Meier empirical curve; the naive fit "
                 "treats capped claims as exact and falls away in the tail.",
                 survival_chart(ctx["series"]),
                 _legend([("Empirical (Kaplan&ndash;Meier)", "ln", "border-color:var(--ink3)"),
                          ("Censoring-aware fit", "ln", "border-color:var(--blue)"),
                          ("Naive fit", "ln", "border-color:var(--orange);border-top-style:dotted")]))

    layer_card = _card("%s layer, per claim" % layer,
                       "Model price %s. The %d%% bootstrap interval spans %s to %s and %s the true price." % (
                           _money(price["model_cost_per_claim"]), int(100 * ci["level"]),
                           _money(ci["ci_low"]), _money(ci["ci_high"]), "contains" if covers else "misses"),
                       layer_chart(price, ci, ctx["true_cost"]),
                       _legend([("Model", "", "background:var(--blue)"),
                                ("Empirical check, %s exposed claims (%.1f%% apart)" % (
                                    "{:,}".format(price["empirical_n"]), abs(price["rel_error_pct"])),
                                 "", "background:var(--orange)"),
                                ("True price", "ln", "border-color:var(--ink)")]))

    prices = ci["prices"]
    pmin, pmax = min(prices), max(prices)
    bins = 10
    bw = (pmax - pmin) / bins or 1.0
    bcounts = [0] * bins
    for p in prices:
        bcounts[min(int((p - pmin) / bw), bins - 1)] += 1
    bedges = [pmin + bw * i for i in range(bins + 1)]
    boot = _card("Bootstrap distribution", "%d refits on resampled claims; each bar counts price estimates." % len(prices),
                 hist_chart(bcounts, bedges, "b0", xfmt=lambda v: "$%d" % round(v),
                            markers=[(ctx["true_cost"], "True")] if pmin <= ctx["true_cost"] <= pmax else [],
                            tip_fmt=lambda i, c: "$%d&ndash;$%d<br><b>%d estimates</b>" % (
                                round(bedges[i]), round(bedges[i + 1]), c), y_label_fmt=lambda v: "%d" % v))

    h = ex["hist"]
    edges = [h["width"] * i for i in range(len(h["counts"]) + 1)]
    limits = sorted(set(r[0] for r in ex["by_limit"]))
    paid = _card("Paid loss distribution",
                 "Spikes appear where claims hit their policy limit; those amounts are lower bounds, not exact losses.",
                 hist_chart(h["counts"], edges, "b0", xfmt=lambda v: "$%dk" % (v // 1000),
                            markers=[(l, "$%dk" % (l // 1000)) for l in limits if l < edges[-1]]))

    cens = _card("Capped claims by policy limit", "Higher limits cap fewer claims, so they reveal more of the tail.",
                 vbar_chart(["$%dk" % (r[0] // 1000) for r in ex["by_limit"]],
                            [("Capped share", [100.0 * r[2] / r[1] for r in ex["by_limit"]], "b0")],
                            fmt=lambda v: "%.0f%%" % v,
                            tip_fmt=lambda c, n, v: "<b>%s limit</b><br>%.1f%% capped" % (c, v)))

    line = _card("Average paid loss by line", "Mean of cleaned, capped paid amounts.",
                 hbar_chart([(r[0], [(r[2], "b0", "Average")]) for r in ex["by_line"]], _money))

    year = _card("Claims by accident year", "Cleaned claim counts after de-duplication.",
                 vbar_chart([str(r[0])[2:] and "'%s" % str(r[0])[2:] for r in ex["by_year"]],
                            [("Claims", [r[1] for r in ex["by_year"]], "b0")], fmt=_short,
                            tip_fmt=lambda c, n, v: "<b>%s</b><br>%s claims" % (c, "{:,}".format(v))))

    perr = _card("Parameter error vs truth", "Censoring-aware EM stays near zero; the naive fit is off by tens of percent.",
                 hbar_chart([("Mix weight", [(pct(f["pi"], tr["pi"]), "b0", "Censoring-aware"),
                                              (pct(nv["pi"], tr["pi"]), "b2", "Naive")]),
                             ("Small-claim mean", [(pct(f["mean_1"], tr["mean_1"]), "b0", "Censoring-aware"),
                                                    (pct(nv["mean_1"], tr["mean_1"]), "b2", "Naive")]),
                             ("Large-claim mean", [(pct(f["mean_2"], tr["mean_2"]), "b0", "Censoring-aware"),
                                                    (pct(nv["mean_2"], tr["mean_2"]), "b2", "Naive")])],
                            lambda v: "%.1f%%" % v),
                 _legend([("Censoring-aware", "", "background:var(--blue)"), ("Naive", "", "background:var(--orange)")]))

    a1, a2 = sel["one_component"], sel["two_component"]
    base = min(a1["aic"], a1["bic"], a2["aic"], a2["bic"])
    ic = _card("Model selection", "Information criteria above the best model (lower is better). "
               "Both prefer two components.",
               vbar_chart(["AIC", "BIC"], [("1 component", [a1["aic"] - base, a1["bic"] - base], "b1"),
                                           ("2 components", [a2["aic"] - base, a2["bic"] - base], "b0")],
                          fmt=_short, tip_fmt=lambda c, n, v: "<b>%s, %s</b><br>+%s above best" % (c, n, _short(v))),
               _legend([("1 component", "", "background:var(--gray)"), ("2 components", "", "background:var(--blue)")]))

    qual = _card("Data quality", "Defects injected into the synthetic table against what the checks found.",
                 hbar_chart([(lbl, [(truth[k], "b1", "Injected"), (q[k], "b0", "Detected")])
                             for lbl, k in [("Null losses", keys[0]), ("Negative losses", keys[1]),
                                            ("Duplicate IDs", keys[2])]], lambda v: "%d" % v),
                 _legend([("Injected", "", "background:var(--gray)"), ("Detected", "", "background:var(--blue)")]))

    rules = _card("Rule retrieval", "BM25 over the rating-rule snippets, scored on %d labelled queries." % ret["n"],
                  hbar_chart([("Recall@1", [(100 * ret["r1"], "b0", "Recall")]),
                              ("Recall@3", [(100 * ret["r3"], "b0", "Recall")])], lambda v: "%.0f%%" % v))

    rows = "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td>%.2f%%</td><td>%.1f%%</td></tr>" % (
        n, fm(t), fm(e), pct(e, t), pct(v, t)) for n, t, e, v, fm in [
        ("Mix weight", tr["pi"], f["pi"], nv["pi"], lambda x: "%.4f" % x),
        ("Small-claim mean", tr["mean_1"], f["mean_1"], nv["mean_1"], _money),
        ("Large-claim mean", tr["mean_2"], f["mean_2"], nv["mean_2"], _money)])
    params = ('<div class="card"><h3>Fitted parameters</h3><p class="s">Two-component mixed exponential, right-censored EM, '
              "%d iterations to converge.</p><table><tr><th>Parameter</th><th>True</th><th>Fitted</th>"
              "<th>Error</th><th>Naive error</th></tr>%s</table></div>" % (f["iterations"], rows))

    facts = [("Claims", "{:,}".format(truth["n_rows"])), ("Layer", layer),
             ("Agent", "%d steps, %d tools" % (ag["steps"], ag["tools"])),
             ("Runtime", "%.1f seconds" % ctx["seconds"]), ("&Delta;AIC", "%.0f" % sel["delta_aic"]),
             ("Bootstrap", "%d refits of %s" % (ci["n_boot"], "{:,}".format(ci["sample_size"])))]
    facts_html = '<dl class="facts">%s</dl>' % "".join("<div><dt>%s</dt><dd>%s</dd></div>" % x for x in facts)

    body = """
<nav><div><b>LossLens</b><a href="#severity">Severity</a><a href="#pricing">Pricing</a>
<a href="#model">Model</a><a href="#data">Data</a></div></nav>
<main>
<header class="hero"><p class="eyebrow">Excess layer analysis</p>
<h1>%s layer, priced within %.1f%% of truth.</h1>
<p class="lede">A right-censored severity model fitted to %s claims, with a bootstrap interval on the price.</p>
<div class="stats">%s</div></header>

<section class="block" id="severity"><h2>Severity</h2>
<p class="kicker">Policy limits cap what a claim reveals. Modelling that explicitly is what keeps the tail honest.</p>
%s<div class="grid2">%s%s</div><div class="grid2">%s%s</div></section>

<section class="block" id="pricing"><h2>Pricing</h2>
<p class="kicker">One number per claim, with the uncertainty around it.</p>
%s<div class="grid2">%s%s</div></section>

<section class="block" id="model"><h2>Model</h2>
<p class="kicker">How the fit compares with the truth and with simpler alternatives.</p>
<div class="grid2" style="margin-top:0">%s%s</div><div style="margin-top:20px">%s</div></section>

<section class="block" id="data"><h2>Data</h2>
<p class="kicker">Checks run before any model is fitted.</p>
<div class="grid2" style="margin-top:0">%s%s</div>
<div class="card" style="margin-top:20px"><h3>Run summary</h3><p class="s">&nbsp;</p>%s</div></section>
<p class="foot">Synthetic claims with known ground truth. Generated by LossLens.</p>
</main><div id="tip" class="tip"></div>""" % (
        layer, err, "{:,}".format(truth["n_rows"]), stats_html,
        surv, paid, cens, line, year, layer_card, boot, _card(
            "Why it matters", "Share of claims above each limit's cap.",
            "<p style='margin:0;color:var(--ink2);font-size:15px'>%.1f%% of claims are capped overall. Ignoring that "
            "puts the large-claim mean off by %.1f%%, against %.2f%% when censoring is modelled.</p>" % (
                100 * f["censored_fraction"], pct(nv["mean_2"], tr["mean_2"]), pct(f["mean_2"], tr["mean_2"]))),
        perr, ic, params, qual, rules, facts_html)
    js = JS % json.dumps({k: [round(v, 6) for v in vs] for k, vs in ctx["series"].items()})
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>LossLens</title><style>%s</style></head><body>%s<script>%s</script></body></html>'
            % (CSS, body, js))
