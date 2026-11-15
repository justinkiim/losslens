"""BM25 retrieval over a small synthetic corpus of generic rating-rule snippets (original text,
not taken from any insurer or bureau manual). Gives the agent retrieval-augmented answers."""
import math
import re

CORPUS = [
    ("R01", "Deductible: the amount of each loss the insured retains before the insurer pays. A per-occurrence deductible applies once to each occurrence regardless of the number of claimants."),
    ("R02", "Policy limit: the maximum amount the insurer pays for a covered loss. Losses above the limit are borne by the insured, so paid losses are right-censored at the limit."),
    ("R03", "Increased limits factor: a multiplier applied to the basic limit premium to price higher limits of liability, reflecting the greater expected loss in the layer above the basic limit."),
    ("R04", "Excess layer pricing: the expected loss in a layer is the difference between limited expected values at the layer top and at the attachment point."),
    ("R05", "Loss development: reported losses on an accident year change as claims mature, so development factors project immature losses to ultimate before rates are reviewed."),
    ("R06", "Trending: historical losses are adjusted for inflation and exposure change to the future policy period before computing loss costs."),
    ("R07", "Experience rating: a risk's own loss history is blended with class expectation using a credibility weight that grows with premium volume and claim count."),
    ("R08", "Class rating: premium is determined by classification, territory, and construction or occupancy factors multiplied against a base rate."),
    ("R09", "Coinsurance clause: the insured must carry insurance to a stated percentage of property value or bear a proportional penalty on partial losses."),
    ("R10", "Occurrence versus claims-made: an occurrence policy covers events during the policy period regardless of report date, while claims-made covers claims reported in the period."),
    ("R11", "Reinsurance treaty: an excess of loss treaty cedes losses above a retention up to a limit, protecting the primary insurer against large single losses."),
    ("R12", "Statistical plan reporting: insurers submit premium and loss transactions by coverage, class, and state so bureaus can validate completeness and consistency of the data."),
    ("R13", "Rate adequacy: indicated rate change compares expected loss ratio at current rates with the permissible loss ratio after expenses and profit loading."),
    ("R14", "Large loss capping: individual losses above a chosen threshold are capped for ratemaking and the excess is spread across classes through an excess loss factor."),
]

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in and or is are for on by as at be it its with that this".split())


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25(object):
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = [tokenize(t) for _, t in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / float(self.N)
        self.df = {}
        for toks in self.toks:
            for w in set(toks):
                self.df[w] = self.df.get(w, 0) + 1

    def _idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query, k=3):
        q = tokenize(query)
        scored = []
        for i, toks in enumerate(self.toks):
            dl = len(toks)
            s = 0.0
            for w in q:
                f = toks.count(w)
                if f:
                    s += self._idf(w) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [{"id": self.docs[i][0], "score": round(s, 3), "text": self.docs[i][1]} for s, i in scored[:k]]


INDEX = BM25(CORPUS)

# (query, expected doc id) pairs used to measure retrieval quality
EVAL_SET = [
    ("what happens to losses above the policy limit", "R02"),
    ("how is the expected loss in an excess layer computed", "R04"),
    ("amount the insured retains before the insurer pays", "R01"),
    ("multiplier to price higher limits of liability", "R03"),
    ("project immature losses to ultimate with development factors", "R05"),
    ("adjust historical losses for inflation before computing loss cost", "R06"),
    ("credibility weight blends own loss history with class expectation", "R07"),
    ("penalty if insured carries less than a percentage of property value", "R09"),
    ("difference between occurrence and claims-made coverage", "R10"),
    ("excess of loss treaty retention protects against large single losses", "R11"),
    ("submit premium and loss transactions to bureaus for validation", "R12"),
    ("indicated rate change versus permissible loss ratio", "R13"),
    ("cap individual large losses and spread the excess across classes", "R14"),
    ("territory and construction factors against a base rate", "R08"),
]


def recall_at_k(k):
    hits = sum(1 for q, d in EVAL_SET if d in [r["id"] for r in INDEX.search(q, k)])
    return hits / float(len(EVAL_SET))
