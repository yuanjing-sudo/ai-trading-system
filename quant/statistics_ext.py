"""Statistical and time-series toolkit for quant research.

Provides correlation/moment analytics, entropy and fractal measures, stationarity
and cointegration tests, distribution fitting, regression helpers, and compact
implementations of standard time-series models (AR, GARCH(1,1), Kalman filter,
Gaussian mixture, k-means, PCA) using nothing but NumPy and pandas.

Everything is implemented from first principles so the package remains fully
offline-capable; statistical p-values use closed-form or standard
approximations and are documented where approximate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant.indicators import _as_series

__all__ = [
    # scalar statistics
    "pearson", "spearman", "kendall_tau", "rank_ic", "covariance",
    "mad", "skew_kurt_ratio", "jarque_bera", "norm_cdf", "chi2_sf",
    "fit_normal", "fit_t", "fit_lognormal", "fit_laplace", "gpd_fit",
    # rolling statistics
    "rolling_autocorr", "rolling_mad", "rolling_moment", "rolling_skew",
    "rolling_kurt", "rolling_spearman", "rolling_jarque_bera",
    "rolling_hurst", "rolling_regression", "zscore_df", "winsorize",
    "clip_outliers", "percentile_of_score", "detrend_linear",
    "detrend_rolling", "diff_series", "fractional_diff",
    # autocorrelation / stationarity
    "autocorr", "partial_autocorr", "ljung_box", "durbin_watson",
    "variance_ratio", "variance_ratio_z", "half_life", "hurst_dfa",
    "fractal_dimension_higuchi", "fractal_dimension_katz",
    "fractal_dimension_petrosian", "adf_test", "kpss_stat",
    "engle_granger",
    # entropy / information
    "sample_entropy", "approximate_entropy", "permutation_entropy",
    "spectral_entropy", "mutual_information", "transfer_entropy",
    "cross_autocorr", "lead_lag_corr",
    # spectral / filtering
    "periodogram", "fft_dominant_cycles", "savgol_filter",
    "biquad_lowpass", "biquad_highpass", "wavelet_haar", "wavelet_energy",
    # distributions / shrinkage
    "ewma_cov", "ewma_corr", "ledoit_wolf_shrinkage",
    "james_stein_shrink",
    # regression / linear models
    "ols_fit", "ridge_regression", "huber_regression", "polyfit_series",
    "LogisticRegressionLite", "PCA", "KMeans", "GaussianMixture1D",
    # time-series models
    "ar_fit", "ar_predict", "garch11_fit", "ou_process_fit",
    "KalmanFilter1D", "KalmanHedgeRatio", "seasonal_decompose",
    "cusum_changepoints", "binary_segmentation_changepoints",
    # resampling / inference
    "bootstrap_ci", "block_bootstrap", "stationary_bootstrap",
    "permutation_test", "nelder_mead",
]


# ---------------------------------------------------------------------------
# Scalar statistics
# ---------------------------------------------------------------------------

def pearson(x, y) -> float:
    """Pearson correlation coefficient."""
    a = np.asarray(x, dtype="float64")
    b = np.asarray(y, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype="float64")
    ranks[order] = np.arange(1, len(x) + 1, dtype="float64")
    # average ties
    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i + 1, j + 2, dtype="float64"))
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation coefficient."""
    a = np.asarray(x, dtype="float64")
    b = np.asarray(y, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return pearson(_rankdata(a[m]), _rankdata(b[m]))


def kendall_tau(x, y) -> float:
    """Kendall tau-b rank correlation (O(n^2), intended for research-sized data)."""
    a = np.asarray(x, dtype="float64")
    b = np.asarray(y, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 3:
        return float("nan")
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        prod = da * db
        concordant += int((prod > 0).sum())
        discordant += int((prod < 0).sum())
        ties_a += int(((da == 0) & (db != 0)).sum())
        ties_b += int(((db == 0) & (da != 0)).sum())
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return float((concordant - discordant) / denom) if denom else 0.0


def rank_ic(pred, actual) -> float:
    """Rank information coefficient (Spearman correlation)."""
    return spearman(pred, actual)


def covariance(x, y) -> float:
    """Sample covariance."""
    a = np.asarray(x, dtype="float64")
    b = np.asarray(y, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2:
        return float("nan")
    return float(np.cov(a, b, ddof=1)[0, 1])


def mad(x, constant: float = 1.4826) -> float:
    """Median absolute deviation (scaled to be normal-consistent by default)."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    med = np.median(a)
    return float(constant * np.median(np.abs(a - med)))


def skew_kurt_ratio(x) -> float:
    """Skewness divided by excess kurtosis (tail asymmetry balance)."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) < 4:
        return float("nan")
    m = a.mean()
    s = a.std()
    if s == 0:
        return 0.0
    skew = float(np.mean(((a - m) / s) ** 3))
    kurt = float(np.mean(((a - m) / s) ** 4)) - 3.0
    return skew / kurt if kurt != 0 else 0.0


def norm_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function via the Wilson-Hilferty approximation."""
    if df <= 0 or x <= 0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return float(1.0 - norm_cdf(z))


def jarque_bera(x) -> dict:
    """Jarque-Bera normality test. p-value is exact for the chi2(2) limit."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 4:
        return {"statistic": float("nan"), "pvalue": float("nan"), "normal": False}
    m = a.mean()
    s = a.std()
    if s == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "normal": True}
    skew = float(np.mean(((a - m) / s) ** 3))
    kurt = float(np.mean(((a - m) / s) ** 4)) - 3.0
    jb = n / 6.0 * (skew ** 2 + kurt ** 2 / 4.0)
    p = math.exp(-jb / 2.0)
    return {"statistic": float(jb), "pvalue": float(p), "normal": bool(p > 0.05)}


def fit_normal(x) -> tuple[float, float]:
    """Method-of-moments normal fit: (mean, std)."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else (float(a.mean()), 0.0)


def fit_lognormal(x) -> tuple[float, float]:
    """Method-of-moments lognormal fit on positive data: (mu, sigma) of log x."""
    a = np.asarray(x, dtype="float64")
    a = a[(np.isfinite(a)) & (a > 0)]
    if len(a) < 2:
        return (float("nan"), float("nan"))
    logs = np.log(a)
    return float(logs.mean()), float(logs.std(ddof=1))


def fit_laplace(x) -> tuple[float, float]:
    """Method-of-moments Laplace fit: (location, scale)."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return (float("nan"), float("nan"))
    loc = float(np.median(a))
    scale = float(np.mean(np.abs(a - loc)))
    return loc, max(scale, 1e-12)


def fit_t(x) -> tuple[float, float, float]:
    """Rough Student-t fit (kurtosis matching): (df, loc, scale)."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return (float("nan"), float("nan"), float("nan"))
    loc = float(np.mean(a))
    scale = float(np.std(a, ddof=1))
    if scale == 0:
        return (float("nan"), loc, 0.0)
    kurt = float(np.mean(((a - loc) / scale) ** 4))
    if kurt <= 3.2:
        return (100.0, loc, scale)
    df = 6.0 / (kurt - 3.0) + 4.0
    return (float(df), loc, scale * math.sqrt((df - 2.0) / df) if df > 2 else scale)


def gpd_fit(excesses, ) -> tuple[float, float]:
    """Generalized Pareto fit via probability-weighted moments: (xi, beta).

    ``excesses`` are positive exceedances over a high threshold.
    """
    a = np.asarray(excesses, dtype="float64")
    a = a[(np.isfinite(a)) & (a > 0)]
    n = len(a)
    if n < 10:
        return (float("nan"), float("nan"))
    a_sorted = np.sort(a)
    j = np.arange(1, n + 1, dtype="float64")
    w0 = a_sorted.mean()
    w1 = float(np.mean((n - j) / (n - 1) * a_sorted))
    xi = 2.0 - w0 / (w0 - 2.0 * w1) if (w0 - 2.0 * w1) != 0 else 0.0
    beta = 2.0 * w0 * w1 / (w0 - 2.0 * w1) if (w0 - 2.0 * w1) != 0 else w0
    return (float(xi), float(beta))


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_autocorr(series: pd.Series, lag: int = 1, period: int = 60) -> pd.Series:
    """Rolling autocorrelation at the given lag."""
    s = _as_series(series)
    return s.rolling(period, min_periods=period).apply(
        lambda w: float(pd.Series(w).autocorr(lag=lag)) if len(w) > lag else np.nan, raw=False
    )


def rolling_mad(series: pd.Series, period: int = 20, constant: float = 1.4826) -> pd.Series:
    """Rolling median absolute deviation."""
    s = _as_series(series)
    med = s.rolling(period, min_periods=period).median()
    return (s - med).abs().rolling(period, min_periods=period).median() * constant


def rolling_moment(series: pd.Series, order: int = 3, period: int = 60) -> pd.Series:
    """Rolling central moment of the given order."""
    s = _as_series(series)

    def _moment(w: np.ndarray) -> float:
        m = w.mean()
        return float(np.mean((w - m) ** order))

    return s.rolling(period, min_periods=period).apply(_moment, raw=True)


def rolling_skew(series: pd.Series, period: int = 60) -> pd.Series:
    """Rolling skewness."""
    return _as_series(series).rolling(period, min_periods=period).skew()


def rolling_kurt(series: pd.Series, period: int = 60) -> pd.Series:
    """Rolling excess kurtosis."""
    return _as_series(series).rolling(period, min_periods=period).kurt()


def rolling_spearman(a: pd.Series, b: pd.Series, period: int = 60) -> pd.Series:
    """Rolling Spearman correlation."""
    sa = _as_series(a)
    sb = _as_series(b).reindex_like(sa)
    va = sa.to_numpy(dtype="float64")
    vb = sb.to_numpy(dtype="float64")
    out = np.full(len(va), np.nan, dtype="float64")
    for i in range(period - 1, len(va)):
        wa = va[i - period + 1:i + 1]
        wb = vb[i - period + 1:i + 1]
        if np.isfinite(wa).all() and np.isfinite(wb).all():
            out[i] = spearman(wa, wb)
    return pd.Series(out, index=sa.index)


def rolling_jarque_bera(series: pd.Series, period: int = 60) -> pd.Series:
    """Rolling Jarque-Bera statistic."""
    return _as_series(series).rolling(period, min_periods=period).apply(
        lambda w: jarque_bera(w)["statistic"], raw=True
    )


def rolling_hurst(series: pd.Series, period: int = 250, min_window: int = 8) -> pd.Series:
    """Rolling Hurst exponent via rescaled range (R/S)."""
    s = _as_series(series)

    def _rs(w: np.ndarray) -> float:
        w = w[np.isfinite(w)]
        n = len(w)
        if n < min_window * 2:
            return np.nan
        windows = np.unique(np.linspace(min_window, n // 2, 5).astype(int))
        logs = []
        for win in windows:
            rs_vals = []
            for start in range(0, n - win + 1, win):
                seg = w[start:start + win]
                mean = seg.mean()
                dev = np.cumsum(seg - mean)
                r = dev.max() - dev.min()
                sd = seg.std()
                if sd > 0:
                    rs_vals.append(r / sd)
            if rs_vals:
                logs.append((math.log(win), math.log(np.mean(rs_vals))))
        if len(logs) < 2:
            return np.nan
        xs = np.array([p[0] for p in logs])
        ys = np.array([p[1] for p in logs])
        return float(np.polyfit(xs, ys, 1)[0])

    return s.rolling(period, min_periods=period).apply(_rs, raw=True)


def rolling_regression(y: pd.Series, x: pd.Series, period: int = 60) -> pd.DataFrame:
    """Rolling OLS of y on x; returns beta, alpha, r2, residual std."""
    sy = _as_series(y)
    sx = _as_series(x).reindex_like(sy)
    vy = sy.to_numpy(dtype="float64")
    vx = sx.to_numpy(dtype="float64")
    n = len(vy)
    beta = np.full(n, np.nan, dtype="float64")
    alpha = np.full(n, np.nan, dtype="float64")
    r2 = np.full(n, np.nan, dtype="float64")
    resid_std = np.full(n, np.nan, dtype="float64")
    for i in range(period - 1, n):
        yy = vy[i - period + 1:i + 1]
        xx = vx[i - period + 1:i + 1]
        m = np.isfinite(yy) & np.isfinite(xx)
        yy, xx = yy[m], xx[m]
        if len(yy) < 3 or xx.std() == 0:
            continue
        coef = np.polyfit(xx, yy, 1)
        pred = np.polyval(coef, xx)
        ss_res = float(((yy - pred) ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        beta[i] = coef[0]
        alpha[i] = coef[1]
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        resid_std[i] = math.sqrt(ss_res / max(len(yy) - 2, 1))
    return pd.DataFrame({"beta": beta, "alpha": alpha, "r2": r2, "resid_std": resid_std}, index=sy.index)


def zscore_df(df: pd.DataFrame, period: int = 60) -> pd.DataFrame:
    """Cross-time z-score of every column."""
    mean = df.rolling(period, min_periods=period).mean()
    std = df.rolling(period, min_periods=period).std()
    return (df - mean) / std.replace(0, np.nan)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series at empirical quantiles."""
    s = _as_series(series)
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def clip_outliers(series: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Clip values beyond ``n_std`` standard deviations from the rolling mean."""
    s = _as_series(series)
    mean = s.rolling(20, min_periods=5).mean()
    std = s.rolling(20, min_periods=5).std()
    return s.clip(mean - n_std * std, mean + n_std * std)


def percentile_of_score(series: pd.Series, score: float) -> float:
    """Percentile (0..100) of ``score`` within ``series``."""
    s = _as_series(series).dropna()
    if len(s) == 0:
        return float("nan")
    return float((s <= score).mean() * 100.0)


def detrend_linear(series: pd.Series) -> pd.Series:
    """Remove a least-squares linear trend."""
    s = _as_series(series)
    x = np.arange(len(s), dtype="float64")
    m = s.notna().to_numpy()
    if m.sum() < 2:
        return s
    coef = np.polyfit(x[m], s.to_numpy()[m], 1)
    trend = np.polyval(coef, x)
    return s - trend


def detrend_rolling(series: pd.Series, period: int = 60) -> pd.Series:
    """Remove a rolling-mean trend."""
    s = _as_series(series)
    return s - s.rolling(period, min_periods=max(period // 4, 2)).mean()


def diff_series(series: pd.Series, order: int = 1) -> pd.Series:
    """Apply differencing ``order`` times."""
    s = _as_series(series)
    for _ in range(max(int(order), 0)):
        s = s.diff()
    return s


def fractional_diff(series: pd.Series, d: float = 0.4, threshold: float = 1e-4) -> pd.Series:
    """Fractional differencing with fixed-width weights (Lopez de Prado).

    Produces a stationary series while retaining long memory. ``d`` in (0, 1)
    is the differencing order; weights below ``threshold`` are truncated.
    """
    s = _as_series(series)
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if k > 1000:
            break
    weights_arr = np.array(weights[::-1], dtype="float64")
    width = len(weights_arr)
    values = s.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan, dtype="float64")
    for i in range(width - 1, len(values)):
        window = values[i - width + 1:i + 1]
        if np.isfinite(window).all():
            out[i] = float(np.dot(weights_arr, window))
    return pd.Series(out, index=s.index, name=s.name)


# ---------------------------------------------------------------------------
# Autocorrelation / stationarity
# ---------------------------------------------------------------------------

def autocorr(series, lag: int = 1) -> float:
    """Autocorrelation at ``lag``."""
    s = _as_series(series).dropna()
    if len(s) <= lag + 1:
        return float("nan")
    return float(s.autocorr(lag=lag))


def partial_autocorr(series, nlags: int = 10) -> np.ndarray:
    """Partial autocorrelations via the Durbin-Levinson recursion."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    s = s - s.mean()
    n = len(s)
    if n < nlags + 2:
        nlags = max(n - 2, 0)
    denom = float(np.dot(s, s))
    if denom == 0:
        return np.zeros(nlags)
    acf = np.array([float(np.dot(s[:-k] if k else s, s[k:]) / denom) for k in range(nlags + 1)])
    pacf = np.zeros(nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    pacf[0] = acf[1]
    phi[1, 1] = acf[1]
    for k in range(2, nlags + 1):
        num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
        phi[k, k] = num / den if den != 0 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf[k - 1] = phi[k, k]
    return pacf


def ljung_box(series, lags: int = 10) -> dict:
    """Ljung-Box test for serial correlation; p-value via Wilson-Hilferty chi2."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n <= lags + 2:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    acf_vals = [autocorr(s, k) for k in range(1, lags + 1)]
    stat = n * (n + 2) * sum((acf_vals[k] ** 2) / max(n - k - 1, 1) for k in range(lags))
    return {"statistic": float(stat), "pvalue": chi2_sf(float(stat), lags)}


def durbin_watson(residuals) -> float:
    """Durbin-Watson statistic (2.0 = no autocorrelation)."""
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    denom = float(np.dot(r, r))
    if denom == 0:
        return 2.0
    return float(np.dot(np.diff(r), np.diff(r)) / denom)


def variance_ratio(series, q: int = 2) -> float:
    """Lo-MacKinlay variance ratio: VR(q) = Var(q-period returns) / (q * Var(1-period))."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    if len(s) < q * 3:
        return float("nan")
    r1 = np.diff(s)
    rq = s[q:] - s[:-q]
    var1 = float(np.var(r1, ddof=1))
    if var1 == 0:
        return float("nan")
    return float(np.var(rq, ddof=1) / (q * var1))


def variance_ratio_z(series, q: int = 2) -> dict:
    """Variance ratio with the homoskedasticity-robust z-statistic."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < q * 4:
        return {"vr": float("nan"), "z": float("nan"), "pvalue": float("nan")}
    vr = variance_ratio(s, q)
    phi = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n)
    z = (vr - 1.0) / math.sqrt(phi) if phi > 0 else float("nan")
    p = 2.0 * (1.0 - norm_cdf(abs(z))) if math.isfinite(z) else float("nan")
    return {"vr": float(vr), "z": float(z), "pvalue": float(p)}


def half_life(series) -> float:
    """Mean-reversion half-life from an AR(1) fit of the series on its lag."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    if len(s) < 20:
        return float("nan")
    y = s[1:]
    x = s[:-1]
    xm, ym = x.mean(), y.mean()
    denom = float(np.dot(x - xm, x - xm))
    if denom == 0:
        return float("nan")
    beta = float(np.dot(x - xm, y - ym) / denom)
    if beta <= 0 or beta >= 1:
        return float("nan") if beta >= 1 else 0.0
    return float(-math.log(2.0) / math.log(beta))


def hurst_dfa(series, scales: tuple[int, ...] = (8, 16, 32, 64, 128)) -> float:
    """Hurst exponent via detrended fluctuation analysis."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < 64:
        return float("nan")
    y = np.cumsum(s - s.mean())
    logs = []
    for scale in scales:
        if scale >= n:
            continue
        segments = n // scale
        if segments < 2:
            continue
        t = np.arange(scale, dtype="float64")
        fluct = []
        for i in range(segments):
            seg = y[i * scale:(i + 1) * scale]
            coef = np.polyfit(t, seg, 1)
            resid = seg - np.polyval(coef, t)
            fluct.append(float(np.sqrt(np.mean(resid ** 2))))
        f = float(np.mean(fluct))
        if f > 0:
            logs.append((math.log(scale), math.log(f)))
    if len(logs) < 2:
        return float("nan")
    xs = np.array([p[0] for p in logs])
    ys = np.array([p[1] for p in logs])
    return float(np.polyfit(xs, ys, 1)[0])


def fractal_dimension_higuchi(series, kmax: int = 10) -> float:
    """Higuchi fractal dimension estimate."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < kmax * 2:
        return float("nan")
    lk = []
    ks = np.arange(1, kmax + 1)
    for k in ks:
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            seg = s[idx]
            length = float(np.sum(np.abs(np.diff(seg))) * (n - 1) / (len(seg) * k))
            lm.append(length)
        lk.append(float(np.mean(lm)) if lm else np.nan)
    lk = np.array(lk)
    mask = np.isfinite(lk) & (lk > 0)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(1.0 / ks[mask]), np.log(lk[mask]), 1)[0]
    return float(slope)


def fractal_dimension_katz(series) -> float:
    """Katz fractal dimension estimate (uses the mean of the series length)."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < 3:
        return float("nan")
    length = float(np.sum(np.abs(np.diff(s))))
    mean_dist = float(np.mean(np.abs(s - s[0])))
    if length == 0 or mean_dist == 0:
        return float("nan")
    d = math.log10(n - 1)
    return float(math.log10(length / mean_dist) / (d + math.log10(length / mean_dist))) if (d + math.log10(length / mean_dist)) != 0 else float("nan")


def fractal_dimension_petrosian(series) -> float:
    """Petrosian fractal dimension estimate from sign changes."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < 4:
        return float("nan")
    diff = np.diff(s)
    sign_changes = int(np.sum(diff[1:] * diff[:-1] < 0))
    if sign_changes + n <= 0:
        return float("nan")
    return float(math.log10(n) / (math.log10(n) + math.log10(n / (n + 0.4 * sign_changes))))


def adf_test(series, max_lag: int = 1) -> dict:
    """Augmented Dickey-Fuller test statistic (constant, no trend).

    Critical values are the standard MacKinnon approximations: -3.43 (1%),
    -2.86 (5%), -2.57 (10%). The p-value is a coarse linear approximation.
    """
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < max_lag + 10:
        return {"statistic": float("nan"), "critical": {}, "stationary": False}
    y = np.diff(s)
    lagged = s[:-1]
    cols = [lagged]
    for lag in range(1, max_lag + 1):
        cols.append(np.concatenate([np.full(lag, np.nan), y[:-lag]]))
    x = np.column_stack(cols)
    y_dep = y
    mask = np.isfinite(y_dep) & np.isfinite(x).all(axis=1)
    y_dep, x = y_dep[mask], x[mask]
    x = np.column_stack([np.ones(len(x)), x])
    try:
        coef, *_ = np.linalg.lstsq(x, y_dep, rcond=None)
    except np.linalg.LinAlgError:
        return {"statistic": float("nan"), "critical": {}, "stationary": False}
    resid = y_dep - x @ coef
    dof = max(len(y_dep) - x.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        return {"statistic": float("nan"), "critical": {}, "stationary": False}
    se = math.sqrt(max(sigma2 * xtx_inv[1, 1], 1e-18))
    stat = float(coef[1] / se)
    # coarse p-value mapping over the standard critical values
    if stat <= -3.43:
        p = 0.01
    elif stat <= -2.86:
        p = 0.05
    elif stat <= -2.57:
        p = 0.10
    elif stat <= -1.62:
        p = 0.30
    else:
        p = 0.60
    return {
        "statistic": stat,
        "pvalue": p,
        "critical": {"1%": -3.43, "5%": -2.86, "10%": -2.57},
        "stationary": bool(stat <= -2.86),
    }


def kpss_stat(series, lags: int = 5) -> float:
    """KPSS stationarity test statistic (level-stationary null)."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < lags + 10:
        return float("nan")
    resid = s - s.mean()
    cum = np.cumsum(resid)
    eta = float(np.sum(cum ** 2) / n ** 2)
    gamma0 = float(np.mean(resid ** 2))
    s2 = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s2 += 2.0 * w * float(np.mean(resid[k:] * resid[:-k]))
    return float(eta / s2) if s2 != 0 else float("nan")


def engle_granger(y, x) -> dict:
    """Engle-Granger cointegration: OLS beta plus ADF test on the residuals."""
    sy = _as_series(y)
    sx = _as_series(x).reindex_like(sy)
    df = pd.concat([sy, sx], axis=1).dropna()
    if len(df) < 30:
        return {"beta": float("nan"), "alpha": float("nan"), "adf": {}, "cointegrated": False}
    yy = df.iloc[:, 0].to_numpy(dtype="float64")
    xx = df.iloc[:, 1].to_numpy(dtype="float64")
    beta, alpha = np.polyfit(xx, yy, 1)
    resid = yy - (alpha + beta * xx)
    adf = adf_test(resid)
    return {
        "beta": float(beta),
        "alpha": float(alpha),
        "residuals": pd.Series(resid, index=df.index),
        "adf": adf,
        "cointegrated": bool(adf.get("stationary", False)),
    }


# ---------------------------------------------------------------------------
# Entropy / information
# ---------------------------------------------------------------------------

def _bin_counts(x: np.ndarray, bins: int = 10) -> np.ndarray:
    counts, _ = np.histogram(x, bins=bins)
    return counts[counts > 0].astype("float64")


def sample_entropy(series, m: int = 2, r: float | None = None) -> float:
    """Sample entropy (Richman & Moorman)."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < m + 2:
        return float("nan")
    if r is None:
        r = 0.2 * float(np.std(s))
    if r <= 0:
        return float("nan")

    def _count(mm: int) -> int:
        templates = np.array([s[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int((dist <= r).sum())
        return count

    b = _count(m)
    a = _count(m + 1)
    if b == 0 or a == 0:
        return float("nan")
    return float(-math.log(a / b))


def approximate_entropy(series, m: int = 2, r: float | None = None) -> float:
    """Approximate entropy (Pincus)."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < m + 2:
        return float("nan")
    if r is None:
        r = 0.2 * float(np.std(s))
    if r <= 0:
        return float("nan")

    def _phi(mm: int) -> float:
        templates = np.array([s[i:i + mm] for i in range(n - mm + 1)])
        counts = []
        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            counts.append(np.mean(dist <= r))
        return float(np.mean(np.log(np.clip(counts, 1e-12, None))))

    return float(_phi(m) - _phi(m + 1))


def permutation_entropy(series, order: int = 3, delay: int = 1) -> float:
    """Normalized permutation entropy in [0, 1]."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < order * delay + 1:
        return float("nan")
    patterns: dict[tuple, int] = {}
    for i in range(n - (order - 1) * delay):
        window = s[i:i + (order - 1) * delay + 1:delay]
        pattern = tuple(np.argsort(window, kind="stable"))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    total = sum(patterns.values())
    probs = np.array(list(patterns.values()), dtype="float64") / total
    ent = -float(np.sum(probs * np.log(probs)))
    return float(ent / math.log(math.factorial(order)))


def spectral_entropy(series, bins: int = 32) -> float:
    """Entropy of the normalized power spectrum in [0, 1]."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    if len(s) < 8:
        return float("nan")
    power = np.abs(np.fft.rfft(s - s.mean())) ** 2
    power = power[1:]
    if power.sum() == 0:
        return float("nan")
    nb = max(min(bins, len(power)), 2)
    folded = power[:len(power) // nb * nb].reshape(-1, nb).sum(axis=0) if len(power) % nb == 0 else power[:nb]
    probs = folded / folded.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)) / math.log(len(probs))) if len(probs) > 1 else 0.0


def mutual_information(x, y, bins: int = 10) -> float:
    """Histogram-based mutual information."""
    a = np.asarray(x, dtype="float64")
    b = np.asarray(y, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < bins * 2:
        return float("nan")
    hist2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist2d / hist2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return float(mi)


def transfer_entropy(x, y, lag: int = 1, bins: int = 5) -> float:
    """Histogram-based transfer entropy from x to y (nats)."""
    a = np.asarray(x, dtype="float64")
    b = np.asarray(y, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < lag + bins * 4:
        return float("nan")
    y_now = b[lag:]
    y_past = b[:-lag]
    x_past = a[:-lag]
    joint = np.stack([y_now, y_past, x_past], axis=1)
    # discretize each channel
    disc = np.zeros_like(joint, dtype=int)
    for c in range(3):
        col = joint[:, c]
        edges = np.quantile(col, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        disc[:, c] = np.clip(np.digitize(col, edges[1:-1]), 0, bins - 1)
    te = 0.0
    n = len(disc)
    data = disc
    # TE = sum p(y2,y1,x1) log( p(y2|y1,x1) / p(y2|y1) )
    joint_counts: dict[tuple, int] = {}
    for row in data:
        joint_counts[tuple(row)] = joint_counts.get(tuple(row), 0) + 1
    y1x1_counts: dict[tuple, int] = {}
    y1_counts: dict[int, int] = {}
    for row in data:
        key = (row[1], row[2])
        y1x1_counts[key] = y1x1_counts.get(key, 0) + 1
        y1_counts[row[1]] = y1_counts.get(row[1], 0) + 1
    for (y2, y1, x1), c in joint_counts.items():
        p_joint = c / n
        p_y2_given_y1x1 = joint_counts[(y2, y1, x1)] / y1x1_counts[(y1, x1)]
        p_y2_given_y1 = 0.0
        for x2 in range(bins):
            k = joint_counts.get((y2, y1, x2))
            if k:
                p_y2_given_y1 += k / y1_counts[y1]
        if p_y2_given_y1 > 0 and p_y2_given_y1x1 > 0:
            te += p_joint * math.log(p_y2_given_y1x1 / p_y2_given_y1)
    return float(te)


def cross_autocorr(x, y, lag: int = 1) -> float:
    """Cross-correlation between x and lagged y."""
    a = np.asarray(x, dtype="float64")
    b = np.asarray(y, dtype="float64")
    if lag >= 0:
        aa, bb = a[lag:], b[:len(b) - lag]
    else:
        aa, bb = a[:len(a) + lag], b[-lag:]
    return pearson(aa, bb)


def lead_lag_corr(x, y, max_lag: int = 10) -> pd.DataFrame:
    """Correlation of x with y shifted over [-max_lag, max_lag]."""
    lags = range(-max_lag, max_lag + 1)
    return pd.DataFrame(
        {"lag": list(lags), "corr": [cross_autocorr(x, y, lag=lag) for lag in lags]}
    )


# ---------------------------------------------------------------------------
# Spectral analysis / digital filters
# ---------------------------------------------------------------------------

def periodogram(series) -> pd.DataFrame:
    """One-sided periodogram: frequency and power columns."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < 4:
        return pd.DataFrame({"frequency": [], "power": []})
    demeaned = s - s.mean()
    power = np.abs(np.fft.rfft(demeaned)) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0)
    return pd.DataFrame({"frequency": freqs, "power": power})


def fft_dominant_cycles(series, top_n: int = 3) -> pd.DataFrame:
    """Strongest spectral cycles as (period, amplitude, power) rows."""
    pg = periodogram(series)
    if pg.empty:
        return pd.DataFrame(columns=["period", "amplitude", "power"])
    pg = pg[(pg["frequency"] > 0) & (pg["power"] > 0)].copy()
    if pg.empty:
        return pd.DataFrame(columns=["period", "amplitude", "power"])
    pg["period"] = 1.0 / pg["frequency"]
    pg["amplitude"] = np.sqrt(pg["power"])
    top = pg.nlargest(top_n, "power")
    return top[["period", "amplitude", "power"]].reset_index(drop=True)


def savgol_filter(series: pd.Series, window: int = 11, poly: int = 3) -> pd.Series:
    """Savitzky-Golay smoothing filter (local polynomial least squares)."""
    s = _as_series(series)
    window = max(int(window) | 1, 3)
    poly = min(int(poly), window - 1)
    half = window // 2
    x = np.arange(-half, half + 1, dtype="float64")
    vander = np.vander(x, poly + 1, increasing=True)
    pinv = np.linalg.pinv(vander)
    weights = pinv[0]
    values = s.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan, dtype="float64")
    for i in range(half, len(values) - half):
        win = values[i - half:i + half + 1]
        if np.isfinite(win).all():
            out[i] = float(np.dot(weights, win))
    return pd.Series(out, index=s.index, name=s.name)


def _biquad(series: pd.Series, b: np.ndarray, a: np.ndarray) -> pd.Series:
    x = series.to_numpy(dtype="float64")
    n = len(x)
    y = np.zeros(n, dtype="float64")
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        xi = x[i] if np.isfinite(x[i]) else 0.0
        yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return pd.Series(y, index=series.index, name=series.name)


def biquad_lowpass(series: pd.Series, cutoff: float = 0.1, q: float = 0.707) -> pd.Series:
    """Second-order Butterworth-style low-pass filter (RBJ biquad).

    ``cutoff`` is normalized to the Nyquist frequency: 0 < cutoff < 1.
    """
    s = _as_series(series)
    w0 = math.pi * float(np.clip(cutoff, 1e-4, 0.999))
    alpha = math.sin(w0) / (2.0 * max(q, 1e-3))
    cos_w0 = math.cos(w0)
    b = np.array([(1.0 - cos_w0) / 2.0, 1.0 - cos_w0, (1.0 - cos_w0) / 2.0])
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    return _biquad(s, b / a[0], a / a[0])


def biquad_highpass(series: pd.Series, cutoff: float = 0.1, q: float = 0.707) -> pd.Series:
    """Second-order high-pass filter (RBJ biquad); ``cutoff`` in (0, 1)."""
    s = _as_series(series)
    w0 = math.pi * float(np.clip(cutoff, 1e-4, 0.999))
    alpha = math.sin(w0) / (2.0 * max(q, 1e-3))
    cos_w0 = math.cos(w0)
    b = np.array([(1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0])
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    return _biquad(s, b / a[0], a / a[0])


def wavelet_haar(series, levels: int = 3) -> dict:
    """Haar discrete wavelet decomposition.

    Returns a dict with the final approximation and one detail array per level.
    """
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    out: dict = {"details": []}
    current = s
    for _ in range(max(int(levels), 0)):
        if len(current) < 2:
            break
        even = current[0::2]
        odd = current[1::2]
        length = min(len(even), len(odd))
        approx = (even[:length] + odd[:length]) / math.sqrt(2.0)
        detail = (even[:length] - odd[:length]) / math.sqrt(2.0)
        out["details"].append(detail)
        current = approx
    out["approximation"] = current
    return out


def wavelet_energy(series, levels: int = 3) -> pd.Series:
    """Energy share of each Haar wavelet detail level plus the approximation."""
    dec = wavelet_haar(series, levels)
    energies = [float(np.sum(d ** 2)) for d in dec["details"]]
    energies.append(float(np.sum(dec["approximation"] ** 2)))
    total = sum(energies) or 1.0
    labels = [f"detail_{i + 1}" for i in range(len(dec["details"]))] + ["approximation"]
    return pd.Series([e / total for e in energies], index=labels)


# ---------------------------------------------------------------------------
# EWMA covariance / shrinkage estimators
# ---------------------------------------------------------------------------

def _as_frame(returns) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        return returns.to_frame()
    return returns


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance matrix of a returns DataFrame."""
    returns = _as_frame(returns)
    x = returns.to_numpy(dtype="float64")
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    n, k = x.shape
    if n < 2:
        return np.full((k, k), np.nan)
    weights = (1.0 - lam) * lam ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()
    mean = np.average(x, axis=0, weights=weights)
    demeaned = x - mean
    return (demeaned * weights[:, None]).T @ demeaned


def ewma_corr(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """Exponentially weighted correlation matrix of a returns DataFrame."""
    returns = _as_frame(returns)
    cov = ewma_cov(returns, lam)
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> dict:
    """Ledoit-Wolf covariance shrinkage toward the scaled identity target.

    Returns the shrunk covariance matrix, the estimated shrinkage intensity
    (0 = sample covariance, 1 = full target), and the target mean variance.
    """
    returns = _as_frame(returns)
    x = returns.to_numpy(dtype="float64")
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    t, n = x.shape
    if t < 2 or n == 0:
        return {"cov": np.full((n, n), np.nan), "shrinkage": float("nan"), "mu": float("nan")}
    x = x - x.mean(axis=0)
    sample = (x.T @ x) / t
    mu = float(np.trace(sample)) / n
    target = mu * np.eye(n)
    x2 = x ** 2
    pi_mat = (x2.T @ x2) / t - sample ** 2
    pi_hat = float(pi_mat.sum())
    gamma_hat = float(((sample - target) ** 2).sum())
    if gamma_hat <= 0:
        delta = 0.0
    else:
        delta = float(np.clip((pi_hat / t) / gamma_hat, 0.0, 1.0))
    shrunk = (1.0 - delta) * sample + delta * target
    return {"cov": shrunk, "shrinkage": delta, "mu": mu}


def james_stein_shrink(x, target: float | None = None) -> dict:
    """James-Stein mean shrinkage toward a common target (default 0)."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    n = len(a)
    m = float(a.mean()) if target is None else float(target)
    if n < 4:
        return {"mean": m, "shrunk": m, "factor": 1.0}
    sigma2 = float(a.var(ddof=1))
    if sigma2 == 0:
        return {"mean": m, "shrunk": m, "factor": 1.0}
    factor = float(np.clip(1.0 - sigma2 / (n * (m - (0.0 if target is None else target)) ** 2 + 1e-12), 0.0, 1.0))
    shrunk = (0.0 if target is None else target) + factor * (m - (0.0 if target is None else target))
    return {"mean": m, "shrunk": float(shrunk), "factor": factor}


# ---------------------------------------------------------------------------
# Regression / linear models
# ---------------------------------------------------------------------------

def ols_fit(X, y) -> dict:
    """OLS with intercept. Accepts 1-D or 2-D ``X``.

    Returns coefficients (without intercept), intercept, r-squared, residuals,
    standard errors, and t-statistics.
    """
    x = np.asarray(X, dtype="float64")
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype="float64").ravel()
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[mask], y[mask]
    n, k = x.shape
    if n <= k + 1:
        return {"coef": np.full(k, np.nan), "intercept": float("nan"), "r2": float("nan"),
                "residuals": np.array([]), "se": np.full(k, np.nan), "tstat": np.full(k, np.nan)}
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = max(n - k - 1, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        cov = np.full((k + 1, k + 1), np.nan)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(resid @ resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = coef / se
    return {
        "coef": coef[1:],
        "intercept": float(coef[0]),
        "r2": float(r2),
        "residuals": resid,
        "se": se[1:],
        "tstat": tstat[1:],
    }


def ridge_regression(X, y, alpha: float = 1.0) -> dict:
    """Ridge regression (closed form, intercept unpenalized)."""
    x = np.asarray(X, dtype="float64")
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype="float64").ravel()
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[mask], y[mask]
    n, k = x.shape
    if n < 2:
        return {"coef": np.full(k, np.nan), "intercept": float("nan")}
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc + alpha * np.eye(k)
    coef = np.linalg.solve(gram, xc.T @ yc)
    intercept = y_mean - float(x_mean @ coef)
    return {"coef": coef, "intercept": intercept}


def huber_regression(X, y, c: float = 1.345, iters: int = 25) -> dict:
    """Huber robust regression via iteratively reweighted least squares."""
    x = np.asarray(X, dtype="float64")
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype="float64").ravel()
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[mask], y[mask]
    n, k = x.shape
    if n <= k + 1:
        return {"coef": np.full(k, np.nan), "intercept": float("nan")}
    design = np.column_stack([np.ones(n), x])
    weights = np.ones(n)
    coef = np.zeros(k + 1)
    for _ in range(max(int(iters), 1)):
        w_sqrt = np.sqrt(weights)
        aw = design * w_sqrt[:, None]
        bw = y * w_sqrt
        coef, *_ = np.linalg.lstsq(aw, bw, rcond=None)
        resid = y - design @ coef
        scale = mad(resid, constant=1.0)
        scale = scale if scale > 0 else float(np.std(resid)) or 1.0
        z = np.abs(resid) / (c * scale)
        weights = np.where(z <= 1.0, 1.0, 1.0 / np.clip(z, 1e-9, None))
    return {"coef": coef[1:], "intercept": float(coef[0]), "weights": weights}


def polyfit_series(series: pd.Series, degree: int = 2) -> np.ndarray:
    """Polynomial coefficients fitting the series against time (highest first)."""
    s = _as_series(series)
    x = np.arange(len(s), dtype="float64")
    mask = s.notna().to_numpy()
    if mask.sum() <= degree:
        return np.full(degree + 1, np.nan)
    return np.polyfit(x[mask], s.to_numpy()[mask], degree)


class LogisticRegressionLite:
    """Binary logistic regression trained with full-batch gradient descent.

    Includes optional L2 regularization and feature standardization for
    numerical stability. No external ML dependencies required.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 500, l2: float = 1e-4, standardize: bool = True):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.standardize = standardize
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self._mean = None
        self._std = None
        self.losses: list[float] = []

    def _prepare(self, X) -> np.ndarray:
        x = np.asarray(X, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        return x

    def fit(self, X, y) -> "LogisticRegressionLite":
        x = self._prepare(X)
        y = np.asarray(y, dtype="float64").ravel()
        mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
        x, y = x[mask], y[mask]
        if self.standardize:
            self._mean = x.mean(axis=0)
            self._std = x.std(axis=0)
            self._std[self._std == 0] = 1.0
            x = (x - self._mean) / self._std
        n, k = x.shape
        self.weights = np.zeros(k)
        self.bias = 0.0
        self.losses = []
        for _ in range(max(int(self.epochs), 1)):
            z = x @ self.weights + self.bias
            z = np.clip(z, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = (x.T @ (p - y)) / n + self.l2 * self.weights
            grad_b = float(np.mean(p - y))
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
            eps = 1e-12
            loss = -float(np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
            loss += 0.5 * self.l2 * float(np.dot(self.weights, self.weights))
            self.losses.append(loss)
        return self

    def predict_proba(self, X) -> np.ndarray:
        x = self._prepare(X)
        if self.standardize and self._mean is not None:
            x = (x - self._mean) / self._std
        z = np.clip(x @ self.weights + self.bias, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PCA:
    """Principal component analysis via SVD (numpy-only)."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components: np.ndarray | None = None
        self.explained_variance: np.ndarray | None = None
        self.explained_variance_ratio: np.ndarray | None = None
        self.mean: np.ndarray | None = None

    def fit(self, X) -> "PCA":
        x = np.asarray(X, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        mask = np.isfinite(x).all(axis=1)
        x = x[mask]
        self.mean = x.mean(axis=0)
        xc = x - self.mean
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        self.components = vt[:k]
        var = (s ** 2) / max(len(x) - 1, 1)
        self.explained_variance = var[:k]
        total = var.sum() or 1.0
        self.explained_variance_ratio = var[:k] / total
        return self

    def transform(self, X) -> np.ndarray:
        x = np.asarray(X, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        return (x - self.mean) @ self.components.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


class KMeans:
    """K-means clustering with k-means++ style seeding (numpy-only)."""

    def __init__(self, n_clusters: int = 3, iters: int = 100, seed: int = 42):
        self.n_clusters = n_clusters
        self.iters = iters
        self.seed = seed
        self.centers: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.inertia = float("nan")

    def _seed_centers(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        centers = [x[rng.integers(len(x))]]
        for _ in range(1, self.n_clusters):
            d2 = np.min([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(x), 1.0 / len(x))
            centers.append(x[rng.choice(len(x), p=probs)])
        return np.array(centers)

    def fit(self, X) -> "KMeans":
        x = np.asarray(X, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        mask = np.isfinite(x).all(axis=1)
        x = x[mask]
        if len(x) < self.n_clusters:
            self.labels = np.zeros(len(x), dtype=int)
            self.centers = x.copy()
            return self
        centers = self._seed_centers(x)
        labels = np.zeros(len(x), dtype=int)
        for _ in range(max(int(self.iters), 1)):
            dists = np.stack([np.sum((x - c) ** 2, axis=1) for c in centers], axis=1)
            labels = dists.argmin(axis=1)
            new_centers = np.array([
                x[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(self.n_clusters)
            ])
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        dists = np.stack([np.sum((x - c) ** 2, axis=1) for c in centers], axis=1)
        self.centers = centers
        self.labels = labels
        self.inertia = float(dists.min(axis=1).sum())
        return self

    def predict(self, X) -> np.ndarray:
        x = np.asarray(X, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        dists = np.stack([np.sum((x - c) ** 2, axis=1) for c in self.centers], axis=1)
        return dists.argmin(axis=1)


class GaussianMixture1D:
    """Univariate Gaussian mixture fitted by expectation-maximization."""

    def __init__(self, n_components: int = 2, iters: int = 200, tol: float = 1e-6, seed: int = 42):
        self.n_components = n_components
        self.iters = iters
        self.tol = tol
        self.seed = seed
        self.means = None
        self.variances = None
        self.weights = None
        self.loglik = float("-inf")

    def _responsibilities(self, x: np.ndarray) -> np.ndarray:
        pdfs = np.stack([
            self.weights[k] * np.exp(-0.5 * (x - self.means[k]) ** 2 / self.variances[k])
            / math.sqrt(2.0 * math.pi * self.variances[k])
            for k in range(self.n_components)
        ], axis=1)
        total = pdfs.sum(axis=1, keepdims=True)
        total[total <= 0] = 1e-300
        return pdfs / total

    def fit(self, X) -> "GaussianMixture1D":
        x = np.asarray(X, dtype="float64").ravel()
        x = x[np.isfinite(x)]
        if len(x) < self.n_components * 2:
            raise ValueError("Not enough observations for Gaussian mixture")
        rng = np.random.default_rng(self.seed)
        quantiles = np.linspace(0.1, 0.9, self.n_components)
        self.means = np.quantile(x, quantiles)
        self.variances = np.full(self.n_components, float(np.var(x)) + 1e-9)
        self.weights = np.full(self.n_components, 1.0 / self.n_components)
        prev_ll = float("-inf")
        for _ in range(max(int(self.iters), 1)):
            resp = self._responsibilities(x)
            nk = resp.sum(axis=0) + 1e-12
            self.means = (resp * x[:, None]).sum(axis=0) / nk
            self.variances = ((resp * (x[:, None] - self.means) ** 2).sum(axis=0) / nk).clip(min=1e-12)
            self.weights = nk / nk.sum()
            weighted_pdfs = self._responsibilities(x)
            ll = float(np.sum(np.log(np.clip(weighted_pdfs.sum(axis=1), 1e-300, None))))
            if abs(ll - prev_ll) < self.tol:
                self.loglik = ll
                break
            prev_ll = ll
            self.loglik = ll
        _ = rng
        return self

    def predict(self, X) -> np.ndarray:
        x = np.asarray(X, dtype="float64").ravel()
        return self._responsibilities(x).argmax(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        x = np.asarray(X, dtype="float64").ravel()
        return self._responsibilities(x)


# ---------------------------------------------------------------------------
# Time-series models
# ---------------------------------------------------------------------------

def nelder_mead(func, x0, maxiter: int = 500, tol: float = 1e-8, step: float = 0.1):
    """Nelder-Mead simplex minimizer. Returns ``(x_best, f_best, n_iter)``."""
    x0 = np.asarray(x0, dtype="float64")
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        point = x0.copy()
        point[i] += step if point[i] == 0 else step * (abs(point[i]) + 0.1)
        simplex.append(point)
    simplex = np.array(simplex)
    fvals = np.array([float(func(p)) for p in simplex])
    for it in range(maxiter):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        if np.max(np.abs(simplex[1:] - simplex[0])) < tol:
            return simplex[0], float(fvals[0]), it
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = float(func(xr))
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = float(func(xe))
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = float(func(xc))
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                fvals[1:] = np.array([float(func(p)) for p in simplex[1:]])
    order = np.argsort(fvals)
    return simplex[order[0]], float(fvals[order[0]]), maxiter


def ar_fit(series, order: int = 1) -> dict:
    """Autoregressive model fit via the Yule-Walker equations.

    Returns ``coef`` (most recent lag first), ``intercept``, and ``sigma2``.
    """
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n <= order + 2:
        return {"coef": np.full(order, np.nan), "intercept": float("nan"), "sigma2": float("nan")}
    mean = float(s.mean())
    centered = s - mean
    denom = float(np.dot(centered, centered))
    if denom == 0:
        return {"coef": np.zeros(order), "intercept": mean, "sigma2": 0.0}
    acov = np.array([float(np.dot(centered[:-k] if k else centered, centered[k:]) / n) for k in range(order + 1)])
    r = acov[1:]
    toeplitz = np.array([[acov[abs(i - j)] for j in range(order)] for i in range(order)])
    try:
        coef = np.linalg.solve(toeplitz, r)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(toeplitz, r, rcond=None)[0]
    sigma2 = float(acov[0] - np.dot(coef, r))
    intercept = mean * (1.0 - float(coef.sum()))
    return {"coef": coef, "intercept": intercept, "sigma2": sigma2}


def ar_predict(series, coef, steps: int = 1, intercept: float = 0.0) -> np.ndarray:
    """Multi-step forecast with an AR model (coef ordered most-recent first)."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    coef = np.asarray(coef, dtype="float64")
    history = list(s[-len(coef):])
    out = []
    for _ in range(max(int(steps), 0)):
        pred = intercept + float(np.dot(coef, history[-len(coef):][::-1]))
        out.append(pred)
        history.append(pred)
    return np.array(out)


def garch11_fit(returns) -> dict:
    """GARCH(1,1) fit by maximum likelihood (Nelder-Mead on the log-likelihood).

    Constraints are enforced through reparameterization:
    ``omega = exp(u0)`` and ``(alpha, beta)`` from a softmax bounded by 0.999,
    guaranteeing a stationary, positive conditional variance.
    """
    r = _as_series(returns).dropna().to_numpy(dtype="float64")
    n = len(r)
    if n < 50:
        return {"omega": float("nan"), "alpha": float("nan"), "beta": float("nan"),
                "loglik": float("nan"), "conditional_vol": None}
    sample_var = float(np.var(r, ddof=1))
    if sample_var <= 0:
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0, "loglik": float("nan"), "conditional_vol": None}

    def neg_loglik(u):
        omega = math.exp(np.clip(u[0], -20, 20))
        e0, e1 = np.exp(u[1]), np.exp(u[2])
        w0 = e0 / (e0 + e1 + 1.0)
        w1 = e1 / (e0 + e1 + 1.0)
        alpha = 0.999 * w0
        beta = 0.999 * w1
        h = sample_var
        ll = 0.0
        for x in r:
            ll += -0.5 * (math.log(h) + (x * x) / h)
            h = omega + alpha * (x * x) + beta * h
        return -ll

    x_best, _, _ = nelder_mead(neg_loglik, np.array([math.log(sample_var * 0.1), 0.0, 1.0]), maxiter=400)
    omega = math.exp(np.clip(x_best[0], -20, 20))
    e0, e1 = np.exp(x_best[1]), np.exp(x_best[2])
    alpha = 0.999 * e0 / (e0 + e1 + 1.0)
    beta = 0.999 * e1 / (e0 + e1 + 1.0)
    h = sample_var
    vols = np.empty(n, dtype="float64")
    for i, x in enumerate(r):
        vols[i] = math.sqrt(max(h, 1e-16))
        h = omega + alpha * (x * x) + beta * h
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(alpha + beta),
        "loglik": float(-neg_loglik(x_best)),
        "conditional_vol": pd.Series(vols, index=_as_series(returns).dropna().index),
    }


def ou_process_fit(spread) -> dict:
    """Ornstein-Uhlenbeck fit via AR(1) on the spread.

    Returns ``theta`` (mean-reversion speed), ``mu`` (long-run mean),
    ``sigma`` (volatility), ``sigma_eq`` (stationary std), and the half-life.
    """
    s = _as_series(spread).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < 20:
        return {"theta": float("nan"), "mu": float("nan"), "sigma": float("nan"),
                "sigma_eq": float("nan"), "half_life": float("nan")}
    x = s[:-1]
    y = s[1:]
    xm, ym = x.mean(), float(y.mean())
    denom = float(np.dot(x - xm, x - xm))
    if denom == 0:
        return {"theta": float("nan"), "mu": float("nan"), "sigma": float("nan"),
                "sigma_eq": float("nan"), "half_life": float("nan")}
    b = float(np.dot(x - xm, y - ym) / denom)
    a = ym - b * xm
    if b <= 0 or b >= 1:
        return {"theta": 0.0, "mu": float("nan"), "sigma": float("nan"),
                "sigma_eq": float("nan"), "half_life": float("nan")}
    theta = -math.log(b)
    mu = a / (1.0 - b)
    resid = y - (a + b * x)
    sigma_eps = float(np.std(resid, ddof=1))
    sigma_eq = sigma_eps / math.sqrt(1.0 - b * b)
    return {
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma_eps),
        "sigma_eq": float(sigma_eq),
        "half_life": float(math.log(2.0) / theta),
    }


class KalmanFilter1D:
    """Random-walk-plus-noise Kalman filter for 1-D series."""

    def __init__(self, process_var: float = 1e-5, measurement_var: float = 1e-3, initial: float | None = None):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.initial = initial

    def filter(self, series) -> pd.DataFrame:
        """Run the filter; returns state, variance, and prediction error."""
        s = _as_series(series)
        z = s.to_numpy(dtype="float64")
        n = len(z)
        state = np.full(n, np.nan)
        var = np.full(n, np.nan)
        innov = np.full(n, np.nan)
        x = self.initial if self.initial is not None else (z[0] if n else 0.0)
        p = 1.0
        for i in range(n):
            p += self.process_var
            if np.isfinite(z[i]):
                k = p / (p + self.measurement_var)
                e = z[i] - x
                x = x + k * e
                p = (1.0 - k) * p
                innov[i] = e
            state[i] = x
            var[i] = p
        return pd.DataFrame({"state": state, "variance": var, "innovation": innov}, index=s.index)


class KalmanHedgeRatio:
    """Time-varying hedge ratio between two series via a Kalman filter.

    State: beta (y ~ beta * x). Tune ``process_var`` for adaptivity and
    ``measurement_var`` for noise tolerance.
    """

    def __init__(self, process_var: float = 1e-5, measurement_var: float = 1e-3, initial_beta: float = 1.0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.initial_beta = initial_beta

    def fit(self, y, x) -> pd.DataFrame:
        """Return beta, spread, and prediction-error series."""
        sy = _as_series(y)
        sx = _as_series(x).reindex_like(sy)
        yy = sy.to_numpy(dtype="float64")
        xx = sx.to_numpy(dtype="float64")
        n = len(yy)
        beta = np.full(n, np.nan)
        spread = np.full(n, np.nan)
        b = self.initial_beta
        p = 1.0
        for i in range(n):
            if not (np.isfinite(yy[i]) and np.isfinite(xx[i])):
                beta[i] = b
                continue
            p += self.process_var
            s = xx[i] * p * xx[i] + self.measurement_var
            k = p * xx[i] / s if s != 0 else 0.0
            resid = yy[i] - b * xx[i]
            b = b + k * resid
            p = (1.0 - k * xx[i]) * p
            beta[i] = b
            spread[i] = resid
        return pd.DataFrame({"beta": beta, "spread": spread}, index=sy.index)


def seasonal_decompose(series, period: int = 5) -> pd.DataFrame:
    """Classical additive decomposition into trend, seasonal, and residual."""
    s = _as_series(series)
    period = max(int(period), 2)
    trend = s.rolling(period, center=True, min_periods=period).mean()
    detrended = s - trend
    phase = np.arange(len(s)) % period
    seasonal_map = pd.Series(detrended.to_numpy()).groupby(phase).mean()
    seasonal = pd.Series(seasonal_map.reindex(phase).to_numpy(), index=s.index)
    resid = s - trend - seasonal
    return pd.DataFrame({"trend": trend, "seasonal": seasonal, "resid": resid}, index=s.index)


def cusum_changepoints(series, threshold: float = 5.0, drift: float = 0.5) -> list[int]:
    """CUSUM change-point detection on a standardized series.

    Returns positional indices where the cumulative sum of standardized
    deviations resets after exceeding ``threshold`` (with ``drift`` slack).
    """
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < 10:
        return []
    mean = s.mean()
    std = s.std()
    if std == 0:
        return []
    z = (s - mean) / std
    pos = neg = 0.0
    points: list[int] = []
    for i in range(n):
        pos = max(0.0, pos + z[i] - drift)
        neg = min(0.0, neg + z[i] + drift)
        if pos > threshold:
            points.append(i)
            pos = 0.0
        elif neg < -threshold:
            points.append(i)
            neg = 0.0
    return points


def binary_segmentation_changepoints(series, max_changepoints: int = 5, min_size: int = 10) -> list[int]:
    """Binary segmentation change-point detection (variance criterion)."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < 2 * min_size:
        return []

    def _cost(a: np.ndarray) -> float:
        if len(a) < 2:
            return 0.0
        return float(np.sum((a - a.mean()) ** 2))

    def _best_split(start: int, end: int):
        seg = s[start:end]
        base = _cost(seg)
        best_gain = 0.0
        best_idx = None
        for cut in range(start + min_size, end - min_size + 1):
            left = s[start:cut]
            right = s[cut:end]
            gain = base - (_cost(left) + _cost(right))
            if gain > best_gain:
                best_gain = gain
                best_idx = cut
        return best_idx, best_gain

    segments = [(0, n)]
    points: list[int] = []
    while len(points) < max_changepoints:
        best_overall = (None, 0.0, None)
        for idx, (start, end) in enumerate(segments):
            cut, gain = _best_split(start, end)
            if cut is not None and gain > best_overall[1]:
                best_overall = (cut, gain, idx)
        cut, gain, seg_idx = best_overall
        if cut is None or gain <= 0:
            break
        start, end = segments.pop(seg_idx)
        segments.append((start, cut))
        segments.append((cut, end))
        points.append(cut)
    return sorted(points)


# ---------------------------------------------------------------------------
# Resampling / inference
# ---------------------------------------------------------------------------

def bootstrap_ci(x, stat_fn=None, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Non-parametric bootstrap confidence interval for a statistic."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return {"stat": float("nan"), "lower": float("nan"), "upper": float("nan")}
    stat_fn = stat_fn or (lambda arr: float(np.mean(arr)))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = a[rng.integers(0, len(a), len(a))]
        stats[i] = float(stat_fn(sample))
    return {
        "stat": float(stat_fn(a)),
        "lower": float(np.quantile(stats, alpha / 2.0)),
        "upper": float(np.quantile(stats, 1.0 - alpha / 2.0)),
        "std": float(stats.std(ddof=1)),
    }


def block_bootstrap(x, block: int = 10, n: int | None = None, seed: int = 42) -> np.ndarray:
    """Moving-block bootstrap preserving short-range dependence."""
    a = np.asarray(x, dtype="float64")
    n = n or len(a)
    block = max(int(block), 1)
    if len(a) < block:
        block = max(len(a), 1)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(len(a) - block, 0) + 1, n_blocks)
    chunks = [a[s:s + block] for s in starts]
    return np.concatenate(chunks)[:n]


def stationary_bootstrap(x, mean_block: float = 10.0, n: int | None = None, seed: int = 42) -> np.ndarray:
    """Stationary bootstrap with geometric block lengths (Politis & Romano)."""
    a = np.asarray(x, dtype="float64")
    n = n or len(a)
    if len(a) == 0:
        return a
    p = 1.0 / max(mean_block, 1.0)
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype="float64")
    idx = int(rng.integers(0, len(a)))
    for i in range(n):
        out[i] = a[idx % len(a)]
        if rng.random() < p:
            idx = int(rng.integers(0, len(a)))
        else:
            idx += 1
    return out


def permutation_test(x, y, stat_fn=None, n: int = 1000, seed: int = 42) -> dict:
    """Two-sided permutation test on a two-sample statistic (mean difference)."""
    a = np.asarray(x, dtype="float64")
    b = np.asarray(y, dtype="float64")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"stat": float("nan"), "pvalue": float("nan")}
    stat_fn = stat_fn or (lambda p, q: float(np.mean(p) - np.mean(q)))
    observed = abs(float(stat_fn(a, b)))
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        perm = rng.permutation(pool)
        stat = abs(float(stat_fn(perm[:len(a)], perm[len(a):])))
        if stat >= observed:
            count += 1
    return {"stat": observed, "pvalue": (count + 1) / (n + 1)}
