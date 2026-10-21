"""Wild-cluster bootstrap inference for high-dimensional fixed-effect models.

`pyfixest`'s own `Feols.wildboottest` builds a dense one-hot matrix of every
fixed effect, which is infeasible for the sensor panel (~5,700 station FE on
~150k samples). By Frisch-Waugh-Lovell, the coefficients and the null-restricted
residuals of the FE model equal those of OLS on the FE-demeaned outcome and
regressors, which pyfixest already stores on the fit (`_Y`, `_X`). The
bootstrap here runs `wildboottest`'s WCR procedure on those demeaned arrays.
This is exact when the FE are nested within clusters (stations within Otto
basins); non-nested FE (biome x year/month) are projected out beforehand, the
standard treatment (as in fwildclusterboot's `fe` argument). The
small-sample adjustment counts only the non-FE regressors in k, so `t_wcb` is
larger than a one-hot design's t by sqrt((N - k_onehot) / (N - k)) (about 2%
on the sensor panel); the bootstrap p-value is unaffected, since that factor
scales the observed and bootstrapped t statistics alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many clusters, Rademacher weights take too few distinct values;
# Webb's six-point distribution is the usual replacement (Webb 2014).
WEBB_WEIGHTS_MAX_CLUSTERS = 12


def wild_cluster_bootstrap(model, param, cluster, reps=9999, seed=0, weights_type=None):
    """Restricted (WCR) wild-cluster bootstrap test of H0: `param` = 0 for a fitted `pyfixest` Feols.

    Returns the analytic t statistic's bootstrap two-sided p-value together
    with the number of clusters and the weights used.
    """
    from wildboottest.wildboottest import WildboottestCL

    # wildboottest's numba kernels need integer cluster ids, not string codes.
    clusters = pd.factorize(model._data[cluster])[0].astype(np.int64)
    if (clusters < 0).any():
        raise ValueError(f"Cluster variable {cluster!r} is missing for some estimation rows.")
    n_clusters = int(clusters.max()) + 1
    if weights_type is None:
        weights_type = "webb" if n_clusters < WEBB_WEIGHTS_MAX_CLUSTERS else "rademacher"

    restriction = np.zeros(model._X.shape[1])
    restriction[list(model._coefnames).index(param)] = 1.0

    boot = WildboottestCL(
        X=model._X,
        Y=np.ravel(model._Y),
        cluster=clusters,
        R=restriction,
        B=reps,
        seed=seed,
        parallel=False,
    )
    boot.get_scores(bootstrap_type="11", impose_null=True, adj=True, cluster_adj=True)
    boot.get_weights(weights_type=weights_type)
    boot.get_numer()
    boot.get_denom()
    boot.get_tboot()
    boot.get_vcov()
    boot.get_tstat()
    boot.get_pvalue(pval_type="two-tailed")
    return pd.Series(
        {
            "Coefficient": param,
            "t_wcb": float(np.ravel(boot.t_stat)[0]),
            "p_wcb": float(np.ravel(boot.pvalue)[0]),
            "n_clusters": n_clusters,
            "weights": weights_type,
            "reps": int(boot.B),
        }
    )
