"""Per-query tell rate of a substitute against a reference, IRIS quant_twin protocol.

Protocol (scripts/quant_twin.py in IRIS-audit, arXiv 2607.20860 App. S10.1):
- Random forest (300 trees, sqrt features, balanced classes) on reference-vs-substitute.
- 40 stratified 50/50 splits. Per split: score s(y) = -log P(reference | y) on the held-out
  half; tau = (1 - alpha) quantile of held-out *reference* scores; q = fraction of held-out
  substitute scores above tau. Report mean q and its 2.5/97.5 percentiles across splits.
- "Clears the floor" if the 2.5th percentile of q is above alpha.
- Also report the realized reference false-alarm rate (should be about alpha by construction)
  and the single-query AUROC of s.

Used both for the Phase 1 baseline reproduction and as detector (c) in Phase 4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class TellRateResult:
    q_mean: float
    q_sd: float
    q_lo: float  # 2.5th percentile across splits
    q_hi: float  # 97.5th percentile across splits
    ref_false_alarm_mean: float
    auroc_mean: float
    n_ref: int
    n_sub: int
    n_features: int
    n_splits: int
    alpha: float

    @property
    def clears_floor(self) -> bool:
        return self.q_lo > self.alpha

    def budget(self, eps: float = 0.1, delta: float = 0.05) -> float:
        """IRIS any-tell budget m* = ln(1/delta) / (eps * q) for routing fraction eps."""
        return float("inf") if self.q_mean <= 0 else float(np.log(1 / delta) / (eps * self.q_mean))

    def as_dict(self) -> dict[str, float | int | bool]:
        d: dict[str, float | int | bool] = dict(asdict(self))
        d["clears_floor"] = self.clears_floor
        d["m_star_eps0.1"] = self.budget()
        return d


def tell_rate(
    x_ref: np.ndarray,
    x_sub: np.ndarray,
    alpha: float = 0.05,
    n_splits: int = 40,
    test_size: float = 0.5,
    seed: int = 20260617,
    n_estimators: int = 300,
) -> TellRateResult:
    x_ref = np.asarray(x_ref, dtype=float)
    x_sub = np.asarray(x_sub, dtype=float)
    if x_ref.ndim != 2 or x_sub.ndim != 2 or x_ref.shape[1] != x_sub.shape[1]:
        raise ValueError("x_ref and x_sub must be 2-D with the same number of columns")
    x = np.vstack([x_ref, x_sub])
    y = np.r_[np.zeros(len(x_ref), dtype=int), np.ones(len(x_sub), dtype=int)]  # 1 = substitute

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    qs, fas, aucs = [], [], []
    for i, (tr, te) in enumerate(splitter.split(x, y)):
        clf = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed + i,
                n_jobs=-1,
            ),
        )
        clf.fit(x[tr], y[tr])
        p_ref = np.clip(clf.predict_proba(x[te])[:, 0], 1e-12, 1.0)
        s = -np.log(p_ref)
        s_ref, s_sub = s[y[te] == 0], s[y[te] == 1]
        tau = np.quantile(s_ref, 1 - alpha)
        qs.append(float(np.mean(s_sub > tau)))
        fas.append(float(np.mean(s_ref > tau)))
        aucs.append(float(roc_auc_score(y[te], s)))

    qa = np.array(qs)
    return TellRateResult(
        q_mean=float(qa.mean()),
        q_sd=float(qa.std(ddof=1)) if len(qa) > 1 else 0.0,
        q_lo=float(np.percentile(qa, 2.5)),
        q_hi=float(np.percentile(qa, 97.5)),
        ref_false_alarm_mean=float(np.mean(fas)),
        auroc_mean=float(np.mean(aucs)),
        n_ref=len(x_ref),
        n_sub=len(x_sub),
        n_features=x.shape[1],
        n_splits=n_splits,
        alpha=alpha,
    )
