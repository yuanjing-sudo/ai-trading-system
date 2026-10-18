# Indicator Reference

Auto-generated from `quant.catalog` - do not edit by hand.
Regenerate with `python scripts/generate_quant_docs.py`.

## Registry summary

| Kind | Count |
| --- | --- |
| candle | 90 |
| chart_pattern | 62 |
| feature | 66 |
| indicator | 262 |
| metric | 81 |
| portfolio | 50 |
| risk | 66 |
| statistic | 87 |
| strategy | 66 |
| **Total** | **830** |

## Close-only indicators (134)

| Indicator | Signature | Description |
| --- | --- | --- |
| `acceleration` | `acceleration(series: 'pd.Series', period: 'int' = 10)` | Change in velocity (second derivative of the smoothed series). |
| `adaptive_rsi` | `adaptive_rsi(series: 'pd.Series', period: 'int' = 10)` | Adaptive RSI (Ehlers): RSI of the adaptive-smoothed momentum path. |
| `adaptive_stochastic` | `adaptive_stochastic(series: 'pd.Series', period: 'int' = 20, smooth: 'int' = 3)` | Stochastic oscillator on the adaptive price path (trend + cycle). |
| `alma` | `alma(series: 'pd.Series', period: 'int' = 9, offset: 'float' = 0.85, sigma: 'float' = 6.0)` | Arnaud Legoux moving average. |
| `apo` | `apo(series: 'pd.Series', fast: 'int' = 12, slow: 'int' = 26)` | Absolute price oscillator: fast EMA minus slow EMA. |
| `bandpass` | `bandpass(series: 'pd.Series', period: 'int' = 20)` | Two-pole bandpass filter centered on ``period`` (Ehlers). |
| `bb_percent_b` | `bb_percent_b(series: 'pd.Series', period: 'int' = 20, num_std: 'float' = 2.0)` | Bollinger %B: position of price within the bands (0 = lower, 1 = upper). |
| `bb_width` | `bb_width(series: 'pd.Series', period: 'int' = 20, num_std: 'float' = 2.0)` | Bollinger band width: (upper - lower) / middle. |
| `bipower_variation` | `bipower_variation(close: 'pd.Series', period: 'int' = 20)` | Bipower variation: jump-robust realized variance estimator. |
| `bollinger_bands` | `bollinger_bands(series: 'pd.Series', period: 'int' = 20, num_std: 'float' = 2.0)` | Bollinger Bands with %B and bandwidth. |
| `center_of_gravity` | `center_of_gravity(series: 'pd.Series', period: 'int' = 10)` | Center of gravity oscillator (Ehlers): weighted price centroid minus price. |
| `cmo` | `cmo(series: 'pd.Series', period: 'int' = 14)` | Chande Momentum Oscillator. |
| `connors_rsi` | `connors_rsi(series: 'pd.Series', rsi_period: 'int' = 3, streak_period: 'int' = 2, rank_period: 'int' = 100)` | Connors RSI: blend of short RSI, streak RSI, and percent rank. |
| `coppock_curve` | `coppock_curve(series: 'pd.Series', roc1: 'int' = 14, roc2: 'int' = 11, wma_period: 'int' = 10)` | Coppock Curve. |
| `cti` | `cti(close: 'pd.Series', period: 'int' = 12)` | Correlation trend indicator: corr(price, time) over ``period`` bars. |
| `curvature` | `curvature(series: 'pd.Series', period: 'int' = 20, step: 'int' = 5)` | Discrete second derivative of a smoothed series. |
| `curve_smoothness` | `curve_smoothness(series: 'pd.Series', period: 'int' = 20)` | Smoothness score in [0, 1]: low path curvature relative to range. |
| `cyber_cycle` | `cyber_cycle(series: 'pd.Series', period: 'int' = 20)` | Ehlers cyber cycle: high-pass filtered, smoothed detrended oscillator. |
| `death_cross` | `death_cross(series: 'pd.Series', fast: 'int' = 50, slow: 'int' = 200)` | Boolean: fast SMA crossed below slow SMA on this bar. |
| `dema` | `dema(series: 'pd.Series', period: 'int' = 20)` | Double exponential moving average. |
| `distance_from_high` | `distance_from_high(close: 'pd.Series', period: 'int' = 252)` | Percent drawdown from the rolling high (<= 0). |
| `distance_from_low` | `distance_from_low(close: 'pd.Series', period: 'int' = 252)` | Percent premium to the rolling low (>= 0). |
| `dominant_cycle` | `dominant_cycle(series: 'pd.Series', min_period: 'int' = 6, max_period: 'int' = 50)` | Dominant cycle period via the autocorrelation periodogram (approximation). |
| `dpo` | `dpo(series: 'pd.Series', period: 'int' = 20)` | Detrended Price Oscillator. |
| `dswma` | `dswma(series: 'pd.Series', period: 'int' = 20)` | Double-smoothed weighted moving average. |
| `dual_differential` | `dual_differential(series: 'pd.Series', period: 'int' = 10)` | Dual differential oscillator: difference of two band-limited paths. |
| `dynamic_momentum_index` | `dynamic_momentum_index(close: 'pd.Series', min_period: 'int' = 5, max_period: 'int' = 30, smooth: 'int' = 14)` | RSI whose lookback adapts to volatility: more speed in calm markets. |
| `efficiency_ratio` | `efficiency_ratio(close: 'pd.Series', period: 'int' = 10)` | Kaufman efficiency ratio: directional move over path length. |
| `ehlers_ema` | `ehlers_ema(series: 'pd.Series', period: 'int' = 10)` | Ehlers one-pole exponential smoother (alpha in [0, 1]). |
| `ehlers_fisher` | `ehlers_fisher(series: 'pd.Series', period: 'int' = 10)` | Fisher transform of the normalized median price (Ehlers' version). |
| `ema` | `ema(series: 'pd.Series', period: 'int' = 20, adjust: 'bool' = False)` | Exponential moving average. |
| `ema_ribbon_score` | `ema_ribbon_score(series: 'pd.Series', periods: 'tuple[int, ...]' = (5, 10, 20, 50, 100))` | Composite trend score in [-1, 1] from the alignment of an EMA ribbon. |
| `envelope` | `envelope(series: 'pd.Series', period: 'int' = 20, pct: 'float' = 0.05)` | Moving average envelope. |
| `even_better_sine` | `even_better_sine(series: 'pd.Series', min_period: 'int' = 6, max_period: 'int' = 50)` | Even-better sine wave: quadrature signal from smoothed Hilbert pairs. |
| `fisher_rsi` | `fisher_rsi(series: 'pd.Series', rsi_period: 'int' = 14, fisher_period: 'int' = 10)` | Fisher transform of RSI: sharpens oscillator turning points. |
| `fractdim` | `fractdim(series: 'pd.Series', period: 'int' = 30)` | Katz fractal dimension. |
| `frama` | `frama(series: 'pd.Series', period: 'int' = 16)` | Fractal adaptive moving average (Ehlers). |
| `gain_loss_ratio` | `gain_loss_ratio(series: 'pd.Series', period: 'int' = 14)` | Average gain divided by average loss (the RS inside RSI). |
| `gap` | `gap(open_: 'pd.Series', close: 'pd.Series')` | Overnight gap in percent: open versus previous close. |
| `gap_volatility` | `gap_volatility(open_: 'pd.Series', close: 'pd.Series', period: 'int' = 20)` | Rolling volatility of overnight gaps. |
| `golden_cross` | `golden_cross(series: 'pd.Series', fast: 'int' = 50, slow: 'int' = 200)` | Boolean: fast SMA crossed above slow SMA on this bar. |
| `guppy_mma` | `guppy_mma(close: 'pd.Series')` | Guppy multiple moving average: short (trader) and long (investor) sets. |
| `hamming_ma` | `hamming_ma(series: 'pd.Series', period: 'int' = 20)` | Hamming-window weighted moving average. |
| `hilbert_transform` | `hilbert_transform(series: 'pd.Series')` | Ehlers' Hilbert transform approximation. |
| `hist_vol_osc` | `hist_vol_osc(close: 'pd.Series', fast: 'int' = 10, slow: 'int' = 30)` | Volatility oscillator: percent spread between short and long realized vol. |
| `hma` | `hma(series: 'pd.Series', period: 'int' = 20)` | Hull moving average. |
| `hurst_exponent` | `hurst_exponent(series: 'pd.Series', max_lag: 'int' = 100)` | Hurst exponent via variance of lagged differences (R/S style estimate). |
| `hurst_rs` | `hurst_rs(series: 'pd.Series', min_window: 'int' = 10, max_window: 'int' = 100, step: 'int' = 10)` | Hurst exponent via classic rescaled range analysis. |
| `imp` | `imp(open_: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Intraday momentum index: RSI-style ratio of body gains to total bodies. |
| `instant_trendline` | `instant_trendline(series: 'pd.Series', period: 'int' = 20)` | Instantaneous trendline: smoothed price with an EMA correction term. |
| `iqr` | `iqr(series: 'pd.Series', period: 'int' = 20)` | Rolling interquartile range (75th minus 25th percentile). |
| `jma_approx` | `jma_approx(series: 'pd.Series', period: 'int' = 14, phase: 'float' = 0.0)` | Approximation of Jurik's adaptive moving average. |
| `jump_variation` | `jump_variation(close: 'pd.Series', period: 'int' = 20)` | Jump variation: realized variance minus bipower variation (floored at 0). |
| `kama` | `kama(series: 'pd.Series', period: 'int' = 10, fast: 'int' = 2, slow: 'int' = 30)` | Kaufman adaptive moving average. |
| `know_sure_thing` | `know_sure_thing(close: 'pd.Series', roc1: 'int' = 10, roc2: 'int' = 15, roc3: 'int' = 20, roc4: 'int' = 30, sma1: 'int' = 10, sma2: 'int' = 10, sma3: 'int' = 10, sma4: 'int' = 15, signal: 'int' = 9)` | Know Sure Thing momentum oscillator. |
| `laguerre_filter` | `laguerre_filter(series: 'pd.Series', gamma: 'float' = 0.7)` | Laguerre four-pole low-pass filter (Ehlers). |
| `laguerre_rsi` | `laguerre_rsi(series: 'pd.Series', gamma: 'float' = 0.7)` | Laguerre RSI: fast, low-lag RSI-style oscillator in [0, 1]. |
| `linreg_channel` | `linreg_channel(series: 'pd.Series', period: 'int' = 20, num_std: 'float' = 2.0)` | Linear regression channel with residual bands. |
| `linreg_forecast` | `linreg_forecast(series: 'pd.Series', period: 'int' = 20, forecast: 'int' = 1)` | One-step-ahead OLS forecast within a rolling window. |
| `linreg_slope` | `linreg_slope(series: 'pd.Series', period: 'int' = 20)` | Rolling OLS slope of price versus time. |
| `lsma` | `lsma(series: 'pd.Series', period: 'int' = 25)` | Least squares moving average (linear regression endpoint). |
| `ma_distance` | `ma_distance(series: 'pd.Series', period: 'int' = 20)` | Percent distance of price from its simple moving average. |
| `ma_ratio` | `ma_ratio(series: 'pd.Series', period: 'int' = 20)` | Ratio of price to its simple moving average. |
| `ma_ribbon` | `ma_ribbon(series: 'pd.Series', periods: 'tuple[int, ...]' = (10, 20, 30, 40, 50, 60))` | DataFrame of simple moving averages across ``periods``. |
| `ma_slope` | `ma_slope(series: 'pd.Series', period: 'int' = 20, lookback: 'int' = 1)` | Normalized slope of a moving average. |
| `macd` | `macd(series: 'pd.Series', fast: 'int' = 12, slow: 'int' = 26, signal: 'int' = 9)` | Moving Average Convergence Divergence. |
| `mama_fama` | `mama_fama(series: 'pd.Series', fast_limit: 'float' = 0.5, slow_limit: 'float' = 0.05)` | MESA adaptive moving average (MAMA) with following adaptive MA (FAMA). |
| `mcginley_dynamic` | `mcginley_dynamic(series: 'pd.Series', period: 'int' = 14, k: 'float' = 0.6)` | McGinley dynamic average: self-adjusting to market speed. |
| `mean_absolute_deviation` | `mean_absolute_deviation(series: 'pd.Series', period: 'int' = 20)` | Mean absolute deviation from the rolling mean. |
| `mid_point` | `mid_point(series: 'pd.Series', period: 'int' = 14)` | Midpoint of the rolling N-bar range. |
| `momentum` | `momentum(series: 'pd.Series', period: 'int' = 10)` | Absolute price momentum: close minus close ``period`` bars ago. |
| `momentum_score` | `momentum_score(close: 'pd.Series', periods: 'tuple[int, ...]' = (5, 10, 21, 63, 126, 252))` | Composite multi-horizon momentum score: mean of rolling z-scored ROC. |
| `noise_cleaner` | `noise_cleaner(series: 'pd.Series', min_period: 'int' = 10, max_period: 'int' = 48)` | Ehlers noise eliminator: adaptive smoother locked to the dominant cycle. |
| `percentile_rank` | `percentile_rank(series: 'pd.Series', period: 'int' = 252)` | Rolling percentile rank in [0, 1]. |
| `pgo` | `pgo(close: 'pd.Series', period: 'int' = 14)` | Pretty good oscillator: deviation of price from its average, ATR-scaled. |
| `phase_accumulator` | `phase_accumulator(series: 'pd.Series', min_period: 'int' = 6, max_period: 'int' = 50)` | Phase accumulator: steady ramp whose rate tracks the dominant cycle. |
| `ppo` | `ppo(series: 'pd.Series', fast: 'int' = 12, slow: 'int' = 26, signal: 'int' = 9)` | Percentage Price Oscillator. |
| `price_above_ma` | `price_above_ma(series: 'pd.Series', period: 'int' = 20)` | Boolean: price above its simple moving average. |
| `price_momentum_percentile` | `price_momentum_percentile(close: 'pd.Series', period: 'int' = 60, lookback: 'int' = 252)` | Percentile rank of N-bar price momentum over a longer lookback. |
| `qstick` | `qstick(open_: 'pd.Series', close: 'pd.Series', period: 'int' = 8)` | Qstick: moving average of candle bodies (close - open). |
| `raised_cosine_ma` | `raised_cosine_ma(series: 'pd.Series', period: 'int' = 20)` | Raised-cosine weighted moving average (tapered edges). |
| `range_position` | `range_position(close: 'pd.Series', period: 'int' = 20)` | Position of price within its rolling range, scaled to [-1, 1]. |
| `realized_semivariance` | `realized_semivariance(close: 'pd.Series', period: 'int' = 20, annualize: 'int' = 252)` | Annualized upside/downside semivariance of returns. |
| `relative_strength` | `relative_strength(close: 'pd.Series', benchmark_close: 'pd.Series', period: 'int' = 14)` | Relative strength versus a benchmark: smoothed ratio of prices. |
| `relative_vigor` | `relative_vigor(close: 'pd.Series', period: 'int' = 10)` | Relative Vigor Index (close-based variant). |
| `renko` | `renko(close: 'pd.Series', brick_size: 'float | None' = None)` | Renko brick series. Brick size defaults to ATR-like mean abs move. |
| `ribbon_width` | `ribbon_width(series: 'pd.Series', periods: 'tuple[int, ...]' = (10, 20, 30, 40, 50, 60))` | Percent spread between the widest and narrowest ribbon averages. |
| `rmi` | `rmi(series: 'pd.Series', period: 'int' = 14, momentum_period: 'int' = 4)` | Relative momentum index: RSI applied to momentum instead of price. |
| `robust_zscore` | `robust_zscore(series: 'pd.Series', period: 'int' = 20)` | Z-score built on median and MAD (outlier resistant). |
| `roc` | `roc(series: 'pd.Series', period: 'int' = 12)` | Rate of change in percent. |
| `roc_ma` | `roc_ma(series: 'pd.Series', roc_period: 'int' = 12, ma_period: 'int' = 5)` | Smoothed rate of change. |
| `roll_spread` | `roll_spread(close: 'pd.Series', period: 'int' = 20)` | Roll's implied bid-ask spread estimator. |
| `rolling_median` | `rolling_median(series: 'pd.Series', period: 'int' = 20)` | Rolling median. |
| `rolling_quantile` | `rolling_quantile(series: 'pd.Series', q: 'float' = 0.5, period: 'int' = 20)` | Rolling quantile at level ``q`` in [0, 1]. |
| `roof` | `roof(series: 'pd.Series', period: 'int' = 20)` | Ehlers roofing filter (high-pass): removes trend, keeps cycle. |
| `rsi` | `rsi(series: 'pd.Series', period: 'int' = 14)` | Wilder's Relative Strength Index. |
| `rsi_ma` | `rsi_ma(series: 'pd.Series', rsi_period: 'int' = 14, ma_period: 'int' = 7)` | Moving average of RSI (smooths the oscillator). |
| `rsi_slope` | `rsi_slope(series: 'pd.Series', period: 'int' = 14, slope: 'int' = 1)` | Bar-over-bar change of RSI. |
| `rvi_volatility` | `rvi_volatility(close: 'pd.Series', period: 'int' = 10, smooth: 'int' = 14)` | Relative volatility index (Dorsey): up-vol share of total volatility. |
| `schaff_trend` | `schaff_trend(close: 'pd.Series', fast: 'int' = 23, slow: 'int' = 50, cycle: 'int' = 10)` | Schaff Trend Cycle. |
| `semideviation` | `semideviation(series: 'pd.Series', period: 'int' = 20, annualize: 'int' = 252)` | Annualized downside deviation of returns. |
| `sine_wave` | `sine_wave(series: 'pd.Series', min_period: 'int' = 6, max_period: 'int' = 50)` | Sine wave indicator: sine/lead-sine pair plus the dominant cycle period. |
| `sma` | `sma(series: 'pd.Series', period: 'int' = 20)` | Simple moving average. |
| `sma_of_sine` | `sma_of_sine(series: 'pd.Series', min_period: 'int' = 6, max_period: 'int' = 50)` | Sine wave with smoothed %D-style average for crossover signals. |
| `sma_oscillator` | `sma_oscillator(series: 'pd.Series', fast: 'int' = 10, slow: 'int' = 20)` | Percent spread between a fast and slow SMA. |
| `smma` | `smma(series: 'pd.Series', period: 'int' = 14)` | Smoothed moving average (Wilder). |
| `stddev` | `stddev(series: 'pd.Series', period: 'int' = 20, ddof: 'int' = 0)` | Rolling standard deviation. |
| `stoch_rsi` | `stoch_rsi(series: 'pd.Series', period: 'int' = 14, k_period: 'int' = 3, d_period: 'int' = 3)` | Stochastic RSI. |
| `stoch_slow_rsi` | `stoch_slow_rsi(series: 'pd.Series', rsi_period: 'int' = 14, stoch_period: 'int' = 14, k: 'int' = 3, d: 'int' = 3)` | Smoothed stochastic RSI with %K and %D lines. |
| `super_smoother` | `super_smoother(series: 'pd.Series', period: 'int' = 10)` | Ehlers two-pole super smoother: near-zero lag with low noise. |
| `t3` | `t3(series: 'pd.Series', period: 'int' = 10, vfactor: 'float' = 0.7)` | T3 moving average (Tillson): triple-smoothed EMA with volume factor. |
| `tema` | `tema(series: 'pd.Series', period: 'int' = 20)` | Triple exponential moving average. |
| `tmo` | `tmo(close: 'pd.Series', long: 'int' = 14, short: 'int' = 5)` | True momentum oscillator (Chande): normalized sum of momentum. |
| `trend_quality` | `trend_quality(close: 'pd.Series', period: 'int' = 20)` | R-squared of a rolling linear fit: how clean the trend is (0..1). |
| `trend_score` | `trend_score(close: 'pd.Series', period: 'int' = 20)` | Composite trend strength: signed slope scaled by R-squared and volatility. |
| `trendflex` | `trendflex(series: 'pd.Series', period: 'int' = 20)` | Trendflex smoothing: minimizes noise while preserving real trend moves. |
| `trima` | `trima(series: 'pd.Series', period: 'int' = 20)` | Triangular moving average (double smoothed SMA). |
| `trimmed_mean` | `trimmed_mean(series: 'pd.Series', period: 'int' = 20, fraction: 'float' = 0.1)` | Rolling trimmed mean dropping ``fraction`` from each tail. |
| `trix` | `trix(series: 'pd.Series', period: 'int' = 15)` | TRIX: 1-day ROC of triple EMA. |
| `tsf` | `tsf(series: 'pd.Series', period: 'int' = 14, offset: 'int' = 0)` | Time series forecast: linear regression projected ``offset`` bars ahead. |
| `tsi` | `tsi(series: 'pd.Series', long: 'int' = 25, short: 'int' = 13)` | True strength index: double-smoothed momentum over its absolute value. |
| `upside_deviation` | `upside_deviation(series: 'pd.Series', period: 'int' = 20, annualize: 'int' = 252)` | Annualized upside deviation of returns. |
| `variance` | `variance(series: 'pd.Series', period: 'int' = 20, ddof: 'int' = 1)` | Rolling variance. |
| `velocity` | `velocity(series: 'pd.Series', period: 'int' = 10)` | Rate of change of a smoothed series (first derivative). |
| `vhf` | `vhf(close: 'pd.Series', period: 'int' = 28)` | Vertical horizontal filter: trending vs ranging markets. |
| `vidya` | `vidya(series: 'pd.Series', period: 'int' = 14, cmo_period: 'int' = 9)` | Variable index dynamic average (Chande): EMA speed follows |CMO|. |
| `vol_of_vol` | `vol_of_vol(close: 'pd.Series', vol_period: 'int' = 20, vov_period: 'int' = 20)` | Volatility of volatility: rolling std of realized volatility. |
| `volatility_ratio` | `volatility_ratio(close: 'pd.Series', short: 'int' = 10, long_: 'int' = 60)` | Ratio of short-window to long-window realized volatility. |
| `winsorized_mean` | `winsorized_mean(series: 'pd.Series', period: 'int' = 20, fraction: 'float' = 0.1)` | Rolling winsorized mean clipping ``fraction`` from each tail. |
| `wma` | `wma(series: 'pd.Series', period: 'int' = 20)` | Weighted moving average (linear weights). |
| `yearly_position` | `yearly_position(close: 'pd.Series', period: 'int' = 252)` | Percentile position within the trailing year's range (0..100). |
| `zero_crossing_rate` | `zero_crossing_rate(series: 'pd.Series', period: 'int' = 20)` | Rate of sign changes of the detrended series over a rolling window. |
| `zlema` | `zlema(series: 'pd.Series', period: 'int' = 20)` | Zero-lag exponential moving average: EMA of de-lagged input. |
| `zscore` | `zscore(series: 'pd.Series', period: 'int' = 20)` | Rolling z-score. |

## OHLC indicators (46)

| Indicator | Signature | Description |
| --- | --- | --- |
| `adr_percent` | `adr_percent(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Average daily range as percent of close. |
| `adx` | `adx(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Average Directional Index. |
| `adxr` | `adxr(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | ADX rating: average of ADX and its value ``period`` bars ago. |
| `atr` | `atr(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Average True Range (Wilder). |
| `atr_bands` | `atr_bands(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 20, mult: 'float' = 2.0)` | SMA middle band with ATR-scaled upper/lower bands. |
| `atr_percent` | `atr_percent(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | ATR as percent of close. |
| `atr_roc` | `atr_roc(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14, roc_period: 'int' = 5)` | Rate of change of ATR: accelerating or decelerating volatility. |
| `bop` | `bop(open_: 'pd.Series', high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', smooth: 'int' = 14)` | Balance of power: (close - open) / (high - low) with SMA smoothing. |
| `camarilla_pivots` | `camarilla_pivots(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Camarilla pivot levels derived from the current bar's HLC. |
| `cci` | `cci(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 20)` | Commodity Channel Index. |
| `choppiness_index` | `choppiness_index(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Choppiness Index: high = choppy, low = trending. |
| `close_location_value` | `close_location_value(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Close location value in [-1, 1]: where the close sits inside the bar. |
| `demark_pivots` | `demark_pivots(open_: 'pd.Series', high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | DeMark pivot points with conditional X calculation. |
| `devstop` | `devstop(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 20, mult: 'float' = 2.0)` | Deviation-scaled trailing stop (Ehlers' dev-stop) with trend state. |
| `dmi` | `dmi(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Directional Movement Index components (+DI, -DI). |
| `dmi_spread` | `dmi_spread(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Plus DI minus minus DI: directional dominance in [-100, 100]. |
| `elder_ray` | `elder_ray(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 13)` | Elder Ray Bull/Bear Power. |
| `fibonacci_pivots` | `fibonacci_pivots(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Fibonacci pivot points (38.2% / 61.8% / 100% levels). |
| `garman_klass_volatility` | `garman_klass_volatility(open_: 'pd.Series', high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 20, annualize: 'int' = 252)` | Garman-Klass volatility estimator. |
| `heikin_ashi` | `heikin_ashi(open_: 'pd.Series', high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Heikin-Ashi smoothed candles. |
| `hl_range_pct` | `hl_range_pct(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Bar range as percent of close. |
| `hlc3` | `hlc3(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Typical price: (high + low + close) / 3. |
| `ichimoku` | `ichimoku(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', tenkan: 'int' = 9, kijun: 'int' = 26, senkou_b: 'int' = 52)` | Ichimoku Kinko Hyo cloud. |
| `kc_width` | `kc_width(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 20, mult: 'float' = 2.0)` | Keltner channel width relative to its middle band. |
| `keltner_channels` | `keltner_channels(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 20, mult: 'float' = 2.0, atr_period: 'int' = 10)` | Keltner Channels based on EMA and ATR. |
| `natr` | `natr(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Normalized ATR in percent. |
| `ohlc4` | `ohlc4(open_: 'pd.Series', high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Average price: (open + high + low + close) / 4. |
| `pivot_points` | `pivot_points(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Classic floor trader pivot points. |
| `range_expansion_index` | `range_expansion_index(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', lookback: 'int' = 5, period: 'int' = 20)` | Ratio of the current true range to its recent average. |
| `rogers_satchell_volatility` | `rogers_satchell_volatility(open_: 'pd.Series', high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 20, annualize: 'int' = 252)` | Rogers-Satchell volatility estimator (drift independent). |
| `rsi_divergence` | `rsi_divergence(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14, pivot_window: 'int' = 3)` | RSI divergence detector: +1 bullish, -1 bearish, 0 elsewhere. |
| `rvi` | `rvi(open_: 'pd.Series', high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 10)` | Relative Vigor Index. |
| `smi` | `smi(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', k: 'int' = 10, d: 'int' = 3, ema_period: 'int' = 3)` | Stochastic momentum index with signal line. |
| `squeeze_momentum` | `squeeze_momentum(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', bb_period: 'int' = 20, bb_mult: 'float' = 2.0, kc_period: 'int' = 20, kc_mult: 'float' = 1.5)` | TTM Squeeze Momentum. |
| `stochastic` | `stochastic(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', k_period: 'int' = 14, d_period: 'int' = 3, smooth_k: 'int' = 3)` | Stochastic oscillator (%K, %D). |
| `supertrend` | `supertrend(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 10, mult: 'float' = 3.0)` | Supertrend indicator with direction. |
| `trend_intensity` | `trend_intensity(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Signed ADX-like intensity in [-1, 1]: trend strength with direction. |
| `true_range` | `true_range(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | True Range (Wilder). |
| `true_range_pct` | `true_range_pct(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | True range as percent of close. |
| `ultimate_oscillator` | `ultimate_oscillator(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', p1: 'int' = 7, p2: 'int' = 14, p3: 'int' = 28)` | Ultimate Oscillator (Williams). |
| `vortex` | `vortex(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Vortex Indicator (+VI, -VI). |
| `wad` | `wad(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Williams accumulation/distribution: bar-by-bar accumulation measure. |
| `weighted_close` | `weighted_close(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Weighted close: (high + low + 2 * close) / 4. |
| `williams_r` | `williams_r(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 14)` | Williams %R. |
| `woodie_pivots` | `woodie_pivots(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series')` | Woodie pivot points (close weighted 2x in the pivot). |
| `yang_zhang_volatility` | `yang_zhang_volatility(open_: 'pd.Series', high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 20, annualize: 'int' = 252)` | Yang-Zhang volatility estimator (handles overnight gaps). |

## OHLCV indicators (37)

| Indicator | Signature | Description |
| --- | --- | --- |
| `ad_line` | `ad_line(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series')` | Accumulation/distribution line (Chaikin): CLV-weighted cumulative volume. |
| `anchored_vwap` | `anchored_vwap(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', anchor_iloc: 'int' = 0)` | Anchored VWAP from a given bar index. |
| `buy_pressure` | `buy_pressure(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', period: 'int' = 13)` | Buying pressure: volume weighted by close position in the bar, summed. |
| `chaikin_oscillator` | `chaikin_oscillator(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', fast: 'int' = 3, slow: 'int' = 10)` | Chaikin Oscillator (EMA of A/D line). |
| `cmf` | `cmf(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', period: 'int' = 20)` | Chaikin Money Flow. |
| `dollar_volume` | `dollar_volume(close: 'pd.Series', volume: 'pd.Series')` | Dollar (turnover) volume. |
| `ease_of_movement` | `ease_of_movement(high: 'pd.Series', low: 'pd.Series', volume: 'pd.Series', period: 'int' = 14)` | Ease of Movement. |
| `force_index` | `force_index(close: 'pd.Series', volume: 'pd.Series', period: 'int' = 13)` | Elder Force Index. |
| `force_ratio` | `force_ratio(close: 'pd.Series', volume: 'pd.Series', period: 'int' = 13)` | Force ratio: smoothed price-change force divided by smoothed volume. |
| `intraday_intensity` | `intraday_intensity(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', period: 'int' = 21)` | Intraday intensity index: closing location weighted by volume, summed. |
| `klinger` | `klinger(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', fast: 'int' = 34, slow: 'int' = 55)` | Klinger Volume Oscillator. |
| `mfi` | `mfi(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', period: 'int' = 14)` | Money Flow Index. |
| `money_flow` | `money_flow(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series')` | Raw money flow (typical price x volume). |
| `negative_volume_index` | `negative_volume_index(close: 'pd.Series', volume: 'pd.Series')` | Negative Volume Index (starts at 1000). |
| `obv` | `obv(close: 'pd.Series', volume: 'pd.Series')` | On-Balance Volume. |
| `obv_ema` | `obv_ema(close: 'pd.Series', volume: 'pd.Series', period: 'int' = 21)` | EMA of on-balance volume. |
| `positive_volume_index` | `positive_volume_index(close: 'pd.Series', volume: 'pd.Series')` | Positive Volume Index (starts at 1000). |
| `pvo` | `pvo(volume: 'pd.Series', fast: 'int' = 12, slow: 'int' = 26, signal: 'int' = 9)` | Percentage Volume Oscillator. |
| `relative_volume` | `relative_volume(volume: 'pd.Series', period: 'int' = 20)` | Volume relative to its simple moving average. |
| `turnover` | `turnover(close: 'pd.Series', volume: 'pd.Series', period: 'int' = 20)` | Average dollar turnover. |
| `turnover_ratio` | `turnover_ratio(volume: 'pd.Series', period: 'int' = 20)` | Raw volume divided by its rolling average (alias of relative volume). |
| `twiggs_money_flow` | `twiggs_money_flow(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', period: 'int' = 21)` | Twiggs money flow: closing location within the true range, volume weighted. |
| `up_down_volume_ratio` | `up_down_volume_ratio(close: 'pd.Series', volume: 'pd.Series', period: 'int' = 50)` | Ratio of up-day volume to down-day volume over a rolling window. |
| `volume_breakout` | `volume_breakout(volume: 'pd.Series', period: 'int' = 20, mult: 'float' = 2.0)` | Boolean: volume above ``mult`` times its rolling average. |
| `volume_dryup` | `volume_dryup(volume: 'pd.Series', period: 'int' = 20, ratio: 'float' = 0.5)` | Boolean: volume below ``ratio`` times its rolling average. |
| `volume_oscillator` | `volume_oscillator(volume: 'pd.Series', fast: 'int' = 5, slow: 'int' = 20)` | Percent spread between fast and slow volume SMAs. |
| `volume_price_corr` | `volume_price_corr(close: 'pd.Series', volume: 'pd.Series', period: 'int' = 20)` | Rolling correlation between price level and volume. |
| `volume_profile` | `volume_profile(close: 'pd.Series', volume: 'pd.Series', bins: 'int' = 20)` | Static volume-at-price histogram. |
| `volume_rsi` | `volume_rsi(volume: 'pd.Series', period: 'int' = 14)` | RSI applied to volume: detects volume thrusts. |
| `volume_trend` | `volume_trend(volume: 'pd.Series', period: 'int' = 20)` | Normalized slope of volume (positive = expanding participation). |
| `volume_zscore` | `volume_zscore(volume: 'pd.Series', period: 'int' = 20)` | Z-score of volume versus its rolling distribution. |
| `vpt` | `vpt(close: 'pd.Series', volume: 'pd.Series')` | Volume price trend: cumulative volume scaled by percent returns. |
| `vwap` | `vwap(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series')` | Cumulative VWAP. |
| `vwema` | `vwema(series: 'pd.Series', volume: 'pd.Series', period: 'int' = 20)` | Volume-weighted exponential moving average. |
| `vwma` | `vwma(series: 'pd.Series', volume: 'pd.Series', period: 'int' = 20)` | Volume weighted moving average. |
| `vwma_typical` | `vwma_typical(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', volume: 'pd.Series', period: 'int' = 20)` | Volume-weighted moving average of the typical price (HLC/3). |
| `vzo` | `vzo(close: 'pd.Series', volume: 'pd.Series', period: 'int' = 14)` | Volume zone oscillator: signed-volume share of total volume. |

## Multi-series / other indicators (45)

| Indicator | Signature | Description |
| --- | --- | --- |
| `accelerator_oscillator` | `accelerator_oscillator(high: 'pd.Series', low: 'pd.Series')` | Bill Williams Accelerator Oscillator. |
| `adr` | `adr(high: 'pd.Series', low: 'pd.Series', period: 'int' = 14)` | Average daily range (absolute). |
| `alpha` | `alpha(asset_returns: 'pd.Series', benchmark_returns: 'pd.Series', period: 'int' = 60, annualize: 'int' = 252)` | Rolling Jensen's alpha (annualized). |
| `amihud_illiquidity` | `amihud_illiquidity(returns: 'pd.Series', dollar_volume: 'pd.Series', period: 'int' = 20)` | Amihud illiquidity: |return| / dollar volume, rolling mean. |
| `aroon` | `aroon(high: 'pd.Series', low: 'pd.Series', period: 'int' = 25)` | Aroon Up/Down. |
| `aroon_oscillator` | `aroon_oscillator(high: 'pd.Series', low: 'pd.Series', period: 'int' = 25)` | Aroon oscillator: Aroon up minus Aroon down, in [-100, 100]. |
| `awesome_oscillator` | `awesome_oscillator(high: 'pd.Series', low: 'pd.Series', fast: 'int' = 5, slow: 'int' = 34)` | Bill Williams Awesome Oscillator. |
| `beta` | `beta(asset_returns: 'pd.Series', benchmark_returns: 'pd.Series', period: 'int' = 60)` | Rolling CAPM beta. |
| `calmar_ratio` | `calmar_ratio(equity: 'pd.Series', annualize: 'int' = 252)` | Calmar ratio: CAGR / |MaxDD|. |
| `chaikin_volatility` | `chaikin_volatility(high: 'pd.Series', low: 'pd.Series', period: 'int' = 10, roc_period: 'int' = 10)` | Chaikin volatility: ROC of the EMA of the high-low range. |
| `crossovers` | `crossovers(a: 'pd.Series', b: 'pd.Series')` | Detect crosses between two lines: +1 golden, -1 death, 0 otherwise. |
| `directional_bias` | `directional_bias(high: 'pd.Series', low: 'pd.Series', period: 'int' = 14)` | Wilder directional bias: net directional movement share in [-1, 1]. |
| `divergence` | `divergence(prices: 'pd.Series', indicator: 'pd.Series', pivot_window: 'int' = 3)` | Generic bullish/bearish divergence detector between price and an indicator. |
| `donchian_channels` | `donchian_channels(high: 'pd.Series', low: 'pd.Series', period: 'int' = 20)` | Donchian Channels. |
| `drawdown_series` | `drawdown_series(equity: 'pd.Series')` | Drawdown series from an equity curve. |
| `entropy` | `entropy(returns: 'pd.Series', bins: 'int' = 10, period: 'int' = 60)` | Rolling Shannon entropy of return distribution. |
| `ewma_volatility` | `ewma_volatility(returns: 'pd.Series', lam: 'float' = 0.94, annualize: 'int' = 252)` | RiskMetrics EWMA volatility. |
| `fib_retracements` | `fib_retracements(high: 'pd.Series', low: 'pd.Series', lookback: 'int' = 100)` | Rolling Fibonacci retracement levels. |
| `fisher_transform` | `fisher_transform(high: 'pd.Series', low: 'pd.Series', period: 'int' = 10)` | Ehlers Fisher Transform. |
| `has_crossed` | `has_crossed(a: 'pd.Series', b: 'pd.Series', lookback: 'int' = 5, direction: 'int' = 1)` | Boolean: a crossed b in the last ``lookback`` bars in ``direction``. |
| `historical_volatility_ratio` | `historical_volatility_ratio(returns: 'pd.Series', short: 'int' = 10, long_: 'int' = 60)` | Ratio of short-term to long-term realized volatility (vol regime signal). |
| `hl2` | `hl2(high: 'pd.Series', low: 'pd.Series')` | Midpoint of high and low. |
| `hl_range` | `hl_range(high: 'pd.Series', low: 'pd.Series')` | Bar range: high minus low. |
| `information_ratio` | `information_ratio(asset_returns: 'pd.Series', benchmark_returns: 'pd.Series', period: 'int' = 60)` | Rolling information ratio. |
| `kurtosis` | `kurtosis(returns: 'pd.Series', period: 'int' = 20)` | Rolling excess kurtosis. |
| `mass_index` | `mass_index(high: 'pd.Series', low: 'pd.Series', fast: 'int' = 9, slow: 'int' = 25)` | Mass Index (Dorsey). |
| `max_drawdown` | `max_drawdown(equity: 'pd.Series')` | Maximum drawdown (negative number). |
| `omega_ratio` | `omega_ratio(returns: 'pd.Series', threshold: 'float' = 0.0, annualize: 'int' = 252)` | Omega ratio at a return threshold. |
| `parabolic_sar` | `parabolic_sar(high: 'pd.Series', low: 'pd.Series', step: 'float' = 0.02, max_step: 'float' = 0.2)` | Parabolic SAR (Wilder). |
| `parkinson_volatility` | `parkinson_volatility(high: 'pd.Series', low: 'pd.Series', period: 'int' = 20, annualize: 'int' = 252)` | Parkinson high-low range volatility estimator. |
| `price_channels` | `price_channels(high: 'pd.Series', low: 'pd.Series', period: 'int' = 20)` | Simple highest-high / lowest-low channels. |
| `realized_volatility` | `realized_volatility(returns: 'pd.Series', period: 'int' = 20, annualize: 'int' = 252)` | Close-to-close realized volatility (annualized by default). |
| `rolling_corr` | `rolling_corr(a: 'pd.Series', b: 'pd.Series', period: 'int' = 60)` | Rolling Pearson correlation. |
| `rolling_cov` | `rolling_cov(a: 'pd.Series', b: 'pd.Series', period: 'int' = 60)` | Rolling covariance. |
| `sharpe_ratio` | `sharpe_ratio(returns: 'pd.Series', rf: 'float' = 0.0, annualize: 'int' = 252)` | Annualized Sharpe ratio. |
| `skewness` | `skewness(returns: 'pd.Series', period: 'int' = 20)` | Rolling skewness. |
| `sortino_ratio` | `sortino_ratio(returns: 'pd.Series', rf: 'float' = 0.0, annualize: 'int' = 252)` | Annualized Sortino ratio (downside deviation). |
| `swing_high` | `swing_high(high: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` | Boolean: bar is a fractal swing high (higher than ``left``/``right`` neighbors). |
| `swing_low` | `swing_low(low: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` | Boolean: bar is a fractal swing low (lower than ``left``/``right`` neighbors). |
| `tracking_error` | `tracking_error(asset_returns: 'pd.Series', benchmark_returns: 'pd.Series', period: 'int' = 60, annualize: 'int' = 252)` | Rolling tracking error of active returns. |
| `true_high` | `true_high(high: 'pd.Series', close: 'pd.Series')` | True high: max(high, previous close). |
| `true_low` | `true_low(low: 'pd.Series', close: 'pd.Series')` | True low: min(low, previous close). |
| `ulcer_index` | `ulcer_index(equity: 'pd.Series')` | Ulcer index (RMS drawdown). |
| `volatility_cone` | `volatility_cone(returns: 'pd.Series', windows=(5, 10, 20, 30, 60, 90, 120, 252), quantiles=(0.1, 0.5, 0.9))` | Volatility cone: realized vol quantiles across lookback windows. |
| `zigzag` | `zigzag(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | ZigZag swing labels: +1 at confirmed peaks, -1 at confirmed troughs. |

## Candlestick patterns (90)

| Pattern | Description |
| --- | --- |
| `abandoned_baby_bear` | Bearish abandoned baby: up bar, gapped-up doji, gapped-down down bar. |
| `abandoned_baby_bull` | Bullish abandoned baby: down bar, gapped-down doji, gapped-up up bar. |
| `advance_block` | Advance block: three rising bars whose bodies shrink (weakening advance). |
| `bearish_counterattack` | Bearish counterattack: gap up then close back at the prior close. |
| `bearish_engulfing` | Bearish engulfing: down body fully engulfs the prior up body after an advance. |
| `bearish_harami` | Bearish harami: small down body contained in the prior up body after an advance. |
| `bearish_harami_cross` | Bearish harami cross: bearish harami whose second bar is a doji. |
| `belt_hold_bear` | Bearish belt hold: opens at the high and runs lower. |
| `belt_hold_bull` | Bullish belt hold: opens at the low and runs higher. |
| `black_marubozu` | Black marubozu: full-length bearish body with no shadows. |
| `breakaway_bear` | Bearish breakaway: gap-up exhaustion relieved by a strong reversing bar. |
| `breakaway_bull` | Bullish breakaway: gap-down exhaustion relieved by a strong reversing bar. |
| `bullish_counterattack` | Bullish counterattack: gap down then close back at the prior close. |
| `bullish_engulfing` | Bullish engulfing: up body fully engulfs the prior down body after a decline. |
| `bullish_harami` | Bullish harami: small up body contained in the prior down body after a decline. |
| `bullish_harami_cross` | Bullish harami cross: bullish harami whose second bar is a doji. |
| `closing_marubozu_black` | Closing marubozu (black): bearish bar closing at the low (no lower shadow). |
| `closing_marubozu_white` | Closing marubozu (white): bullish bar closing at the high (no upper shadow). |
| `concealing_baby_swallow` | Concealing baby swallow: two down bars, an engulfing down bar, then a hammer. |
| `dark_cloud_cover` | Dark cloud cover: up gap open then a close below the midpoint of the prior body. |
| `deliberation` | Deliberation: two long up bars followed by a tiny-bodied star. |
| `doji` | Doji: open and close nearly equal (body <= 10% of range). |
| `down_bar` | Down bar: close below open. |
| `downside_gap_three_methods` | Downside gap three methods: two falling gapped bars, third pulls back into the gap. |
| `dragonfly_doji` | Dragonfly doji: doji opening/closing at the high with a long lower shadow. |
| `evening_doji_star` | Evening doji star: evening star whose middle bar is a doji. |
| `evening_star` | Evening star: long up bar, small gapped-up star, strong down bar. |
| `falling_three_methods` | Falling three methods: long down bar, three contained bounce bars, breakdown bar. |
| `four_price_doji` | Four-price doji: open, high, low, and close all equal. |
| `gravestone_doji` | Gravestone doji: doji at the low with a long upper shadow. |
| `hammer` | Hammer: small body on top of a long lower shadow after a decline. |
| `hanging_man` | Hanging man: hammer shape appearing after an advance. |
| `high_wave` | High-wave bar: unusually wide range with a small body and long shadows. |
| `identical_three_crows` | Identical three crows: three down bars each opening at the prior close. |
| `in_neck` | In-neck: up bar closing just barely above the prior bearish close. |
| `inside_bar` | Inside bar: range contained within the previous bar's range. |
| `inverted_hammer` | Inverted hammer: small body above a long upper shadow after a decline. |
| `kicking_bear` | Bearish kicking: white marubozu followed by a gapped black marubozu. |
| `kicking_bull` | Bullish kicking: black marubozu followed by a gapped white marubozu. |
| `ladder_bottom` | Ladder bottom: three stepped-down bars with small upper shadows, then a gap up. |
| `ladder_top` | Ladder top: three stepped-up bars with small lower shadows, then a gap down. |
| `long_legged_doji` | Long-legged doji: doji with long upper and lower shadows. |
| `long_lower_shadow` | Bullish pin bar: dominant lower shadow, negligible upper shadow. |
| `long_upper_shadow` | Bearish pin bar: dominant upper shadow, negligible lower shadow. |
| `mat_hold_bear` | Bearish mat hold: falling three methods with a gap down on the second bar. |
| `mat_hold_bull` | Bullish mat hold: rising three methods with a gap up on the second bar. |
| `matching_high` | Matching high: two consecutive bullish closes at the same level after an advance. |
| `matching_low` | Matching low: two consecutive bearish closes at the same level after a decline. |
| `meeting_lines_bear` | Bearish meeting lines: long up bar then a long down bar closing at the same price. |
| `meeting_lines_bull` | Bullish meeting lines: long down bar then a long up bar closing at the same price. |
| `morning_doji_star` | Morning doji star: morning star whose middle bar is a doji. |
| `morning_star` | Morning star: long down bar, small gapped-down star, strong up bar. |
| `narrow_range_bar` | Narrow range bar: true range at most ``mult`` times its recent average. |
| `on_neck` | On-neck: up bar closing at (or marginally below) the prior bearish close. |
| `outside_bar` | Outside bar: range engulfs the previous bar's range. |
| `piercing_line` | Piercing line: down gap open then a close above the midpoint of the prior body. |
| `rickshaw_man` | Rickshaw man: long-legged doji with the tiny body centered in the range. |
| `rising_three_methods` | Rising three methods: long up bar, three contained pullback bars, breakout bar. |
| `separating_lines_bear` | Bearish separating lines: up bar then a down bar opening at the same level. |
| `separating_lines_bull` | Bullish separating lines: down bar then an up bar opening at the same level. |
| `shaven_bottom` | Shaven bottom: open equals the bar low (no lower shadow). |
| `shaven_head` | Shaven head: close equals the bar high (no upper shadow). |
| `shooting_star` | Shooting star: inverted-hammer shape appearing after an advance. |
| `side_by_side_white_gapdown` | Downside gap side-by-side white lines: gapped-down pair of near-identical up bars. |
| `side_by_side_white_gapup` | Upside gap side-by-side white lines: gapped-up pair of near-identical up bars. |
| `spinning_top` | Spinning top: small body with shadows on both sides. |
| `stick_sandwich` | Stick sandwich: two identical bearish closes around an up bar. |
| `takuri` | Takuri: dragonfly variant with a small body at the top of the range. |
| `tasuki_gap_down` | Downside tasuki gap: gap down held partially unfilled by the third bar. |
| `tasuki_gap_up` | Upside tasuki gap: gap up held partially unfilled by the third bar. |
| `three_black_crows` | Three black crows: three falling bearish bars with small lower shadows. |
| `three_drives_bear` | Bearish three drives (heuristic): three stair-stepped advances reversing down. |
| `three_drives_bull` | Bullish three drives (heuristic): three stair-stepped declines reversing up. |
| `three_inside_down` | Three inside down: bearish harami followed by a confirming down bar. |
| `three_inside_up` | Three inside up: bullish harami followed by a confirming up bar. |
| `three_line_strike_bear` | Bearish three-line strike: three rising bars erased by one long down bar. |
| `three_line_strike_bull` | Bullish three-line strike: three falling bars erased by one long up bar. |
| `three_outside_down` | Three outside down: bearish engulfing followed by a lower close. |
| `three_outside_up` | Three outside up: bullish engulfing followed by a higher close. |
| `three_stars_in_the_south` | Three stars in the south: three declining dojis, the middle one gapping lower. |
| `three_white_soldiers` | Three white soldiers: three rising bullish bars with small upper shadows. |
| `thrusting` | Thrusting: down bar, gap-down up bar closing into the prior body below its midpoint. |
| `tri_star_bear` | Bearish tri-star: three dojis with the middle star gapping higher. |
| `tri_star_bull` | Bullish tri-star: three dojis with the middle star gapping lower. |
| `two_crows_gapping` | Two crows gapping: white bar, gapped-up black bar, then a lower black bar. |
| `unique_three_river` | Unique three river: down bar, lower-low small down bar, recovering up bar. |
| `up_bar` | Up bar: close above open. |
| `upside_gap_three_methods` | Upside gap three methods: two advancing gapped bars, third pulls back into the gap. |
| `white_marubozu` | White marubozu: full-length bullish body with no shadows. |
| `wide_range_bar` | Wide range bar: true range at least ``mult`` times its recent average. |

## Chart-structure detectors (62)

| Detector | Signature |
| --- | --- |
| `ascending_channel` | `ascending_channel(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` |
| `ascending_triangle` | `ascending_triangle(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, flat: 'float' = 0.02, min_slope: 'float' = 0.015, pct: 'float' = 0.05)` |
| `bear_flag` | `bear_flag(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pole_window: 'int' = 20, flag_window: 'int' = 10, pole_min: 'float' = 0.05)` |
| `bearish_divergence` | `bearish_divergence(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` |
| `break_of_structure` | `break_of_structure(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `breakdown` | `breakdown(close: 'pd.Series', low: 'pd.Series', period: 'int' = 20, confirm_pct: 'float' = 0.0)` |
| `breakout_up` | `breakout_up(close: 'pd.Series', high: 'pd.Series', period: 'int' = 20, confirm_pct: 'float' = 0.0)` |
| `broadening_formation` | `broadening_formation(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` |
| `bull_flag` | `bull_flag(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pole_window: 'int' = 20, flag_window: 'int' = 10, pole_min: 'float' = 0.05)` |
| `bullish_divergence` | `bullish_divergence(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` |
| `change_of_character` | `change_of_character(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `chart_state` | `chart_state(df: 'pd.DataFrame', window: 'int' = 20, pct: 'float' = 0.05)` |
| `cup_and_handle` | `cup_and_handle(close: 'pd.Series', window: 'int' = 80, tol: 'float' = 0.05, handle_frac: 'float' = 0.25, min_depth: 'float' = 0.06)` |
| `descending_channel` | `descending_channel(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` |
| `descending_triangle` | `descending_triangle(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, flat: 'float' = 0.02, min_slope: 'float' = 0.015, pct: 'float' = 0.05)` |
| `detect_patterns` | `detect_patterns(df: 'pd.DataFrame', window: 'int' = 60, tol: 'float' = 0.03, pct: 'float' = 0.05)` |
| `divergence_series` | `divergence_series(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100, mode: 'str' = 'regular')` |
| `double_bottom` | `double_bottom(high: 'pd.Series', low: 'pd.Series', tol: 'float' = 0.03, pct: 'float' = 0.05, min_gap: 'int' = 5)` |
| `double_top` | `double_top(high: 'pd.Series', low: 'pd.Series', tol: 'float' = 0.03, pct: 'float' = 0.05, min_gap: 'int' = 5)` |
| `falling_wedge` | `falling_wedge(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` |
| `false_breakdown` | `false_breakdown(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 20, tol: 'float' = 0.005)` |
| `false_breakout` | `false_breakout(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 20, tol: 'float' = 0.005)` |
| `head_and_shoulders` | `head_and_shoulders(high: 'pd.Series', low: 'pd.Series', shoulder_min: 'float' = 0.02, tol: 'float' = 0.05, pct: 'float' = 0.05)` |
| `hidden_bearish_divergence` | `hidden_bearish_divergence(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` |
| `hidden_bullish_divergence` | `hidden_bullish_divergence(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` |
| `higher_highs` | `higher_highs(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `higher_lows` | `higher_lows(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `horizontal_channel` | `horizontal_channel(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, flat: 'float' = 0.02, pct: 'float' = 0.05)` |
| `inverse_head_and_shoulders` | `inverse_head_and_shoulders(high: 'pd.Series', low: 'pd.Series', shoulder_min: 'float' = 0.02, tol: 'float' = 0.05, pct: 'float' = 0.05)` |
| `level_touches` | `level_touches(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 60, tol: 'float' = 0.01)` |
| `lower_highs` | `lower_highs(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `lower_lows` | `lower_lows(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `macd_divergence` | `macd_divergence(close: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` |
| `obv_divergence` | `obv_divergence(close: 'pd.Series', volume: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` |
| `pattern_frequency` | `pattern_frequency(detections: 'pd.DataFrame')` |
| `pivot_high_values` | `pivot_high_values(high: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` |
| `pivot_highs` | `pivot_highs(high: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` |
| `pivot_low_values` | `pivot_low_values(low: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` |
| `pivot_lows` | `pivot_lows(low: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` |
| `range_position` | `range_position(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 20)` |
| `rectangle_pattern` | `rectangle_pattern(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 40, tol: 'float' = 0.03)` |
| `resistance_distance` | `resistance_distance(close: 'pd.Series', high: 'pd.Series', period: 'int' = 20)` |
| `rising_wedge` | `rising_wedge(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` |
| `rolling_resistance` | `rolling_resistance(high: 'pd.Series', period: 'int' = 20)` |
| `rolling_support` | `rolling_support(low: 'pd.Series', period: 'int' = 20)` |
| `rounding_bottom` | `rounding_bottom(close: 'pd.Series', window: 'int' = 60, min_r2: 'float' = 0.5)` |
| `rounding_top` | `rounding_top(close: 'pd.Series', window: 'int' = 60, min_r2: 'float' = 0.5)` |
| `rsi_divergence` | `rsi_divergence(close: 'pd.Series', period: 'int' = 14, left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` |
| `structure_score` | `structure_score(detections: 'pd.DataFrame', window: 'int' = 5)` |
| `support_distance` | `support_distance(close: 'pd.Series', low: 'pd.Series', period: 'int' = 20)` |
| `swing_count` | `swing_count(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05, kind: 'str' = 'both')` |
| `swing_high_distance` | `swing_high_distance(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `swing_levels` | `swing_levels(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `swing_low_distance` | `swing_low_distance(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `swing_points` | `swing_points(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `swing_series` | `swing_series(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `symmetric_triangle` | `symmetric_triangle(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, flat: 'float' = 0.02, min_slope: 'float' = 0.015, pct: 'float' = 0.05)` |
| `trend_structure` | `trend_structure(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` |
| `triple_bottom` | `triple_bottom(high: 'pd.Series', low: 'pd.Series', tol: 'float' = 0.03, pct: 'float' = 0.05, min_gap: 'int' = 5)` |
| `triple_top` | `triple_top(high: 'pd.Series', low: 'pd.Series', tol: 'float' = 0.03, pct: 'float' = 0.05, min_gap: 'int' = 5)` |
| `v_bottom` | `v_bottom(close: 'pd.Series', window: 'int' = 30, min_move: 'float' = 0.03)` |
| `v_top` | `v_top(close: 'pd.Series', window: 'int' = 30, min_move: 'float' = 0.03)` |

## Strategies (66)

| Strategy | Signature | Description |
| --- | --- | --- |
| `adx_trend` | `adx_trend(df: 'pd.DataFrame', period: 'int' = 14, threshold: 'float' = 25.0)` | Directional trade only when ADX confirms a trend above ``threshold``. |
| `alpha_combo` | `alpha_combo(df: 'pd.DataFrame', weights=None)` | Weighted combination of trend, momentum, reversion, volume, and pattern signals. |
| `aroon_trend` | `aroon_trend(df: 'pd.DataFrame', period: 'int' = 25)` | Aroon trend: long when Aroon-Up leads, short when Aroon-Down leads. |
| `atr_channel_breakout` | `atr_channel_breakout(df: 'pd.DataFrame', period: 'int' = 20, mult: 'float' = 1.5)` | Breakout of an SMA +/- ATR channel. |
| `bollinger_breakout` | `bollinger_breakout(df: 'pd.DataFrame', period: 'int' = 20, mult: 'float' = 2.0)` | Bollinger breakout: enter on band pierce, hold until the mid cross. |
| `bollinger_reversion` | `bollinger_reversion(df: 'pd.DataFrame', period: 'int' = 20, mult: 'float' = 2.0)` | Fade band pierces and exit on the mid-band cross. |
| `bollinger_squeeze_breakout` | `bollinger_squeeze_breakout(df: 'pd.DataFrame', period: 'int' = 20, mult: 'float' = 2.0, lookback: 'int' = 126)` | Enter Bollinger breakouts that occur from a low-width (squeeze) base. |
| `breakout_momentum` | `breakout_momentum(df: 'pd.DataFrame', period: 'int' = 20)` | Enter on a fresh rolling-high breakout, flip on a rolling-low breakdown. |
| `candle_pattern_score` | `candle_pattern_score(df: 'pd.DataFrame', window: 'int' = 5, threshold: 'float' = 0.2)` | Trade the aggregated candlestick pattern score. |
| `cci_reversal` | `cci_reversal(df: 'pd.DataFrame', period: 'int' = 20, lower: 'float' = -100.0, upper: 'float' = 100.0)` | CCI reversal around the +/-100 thresholds. |
| `chart_pattern_structure` | `chart_pattern_structure(df: 'pd.DataFrame', window: 'int' = 60, threshold: 'float' = 0.1)` | Trade the composite chart-formation score. |
| `cmf_trend` | `cmf_trend(df: 'pd.DataFrame', period: 'int' = 20, deadzone: 'float' = 0.0)` | Sign of the Chaikin Money Flow. |
| `connors_rsi` | `connors_rsi(df: 'pd.DataFrame', period: 'int' = 3, lower: 'float' = 10.0, upper: 'float' = 70.0)` | Connors long-only RSI(3) dip buyer. |
| `divergence_strategy` | `divergence_strategy(df: 'pd.DataFrame', period: 'int' = 14, lookback: 'int' = 100, hold: 'int' = 10)` | Hold a position for ``hold`` bars after an RSI divergence confirms. |
| `donchian_breakout` | `donchian_breakout(df: 'pd.DataFrame', period: 'int' = 20)` | Long on a breakout above the rolling high, short below the rolling low. |
| `dual_momentum` | `dual_momentum(df: 'pd.DataFrame', lookback: 'int' = 252, benchmark_df: 'pd.DataFrame | None' = None)` | Absolute momentum with an optional relative leg against a benchmark. |
| `efficiency_trend` | `efficiency_trend(df: 'pd.DataFrame', period: 'int' = 10, threshold: 'float' = 0.3)` | Kaufman efficiency ratio gate around a price/SMA direction. |
| `ema_cross` | `ema_cross(df: 'pd.DataFrame', fast: 'int' = 12, slow: 'int' = 26)` | Long when the fast EMA is above the slow EMA, short otherwise. |
| `engulfing_proxy` | `engulfing_proxy(df: 'pd.DataFrame', threshold: 'float' = 0.2)` | Body-engulfing proxy: reversal candle bodies with opposing follow-through. |
| `ensemble_vote` | `ensemble_vote(df: 'pd.DataFrame', members=None, min_votes: 'int' = 3)` | Majority vote across member strategies: +1/-1 when votes clear the threshold. |
| `faber_timing` | `faber_timing(df: 'pd.DataFrame', period: 'int' = 200)` | Faber-style long-only timing: invested above the 10-month SMA. |
| `gap_reversion` | `gap_reversion(df: 'pd.DataFrame', threshold: 'float' = 0.02)` | Fade overnight gaps and exit when the gap is filled. |
| `hma_trend` | `hma_trend(df: 'pd.DataFrame', period: 'int' = 20, lookback: 'int' = 5)` | Sign of the Hull moving average slope. |
| `ichimoku_trend` | `ichimoku_trend(df: 'pd.DataFrame')` | Ichimoku: long when the conversion line leads the base line, short otherwise. |
| `kama_trend` | `kama_trend(df: 'pd.DataFrame', period: 'int' = 10, lookback: 'int' = 5)` | Sign of the Kaufman adaptive MA slope. |
| `keltner_breakout` | `keltner_breakout(df: 'pd.DataFrame', period: 'int' = 20, mult: 'float' = 2.0)` | Breakout of Keltner channels: long above the upper band, short below. |
| `keltner_reversion` | `keltner_reversion(df: 'pd.DataFrame', period: 'int' = 20, mult: 'float' = 2.0)` | Fade Keltner band pierces and exit at the channel mid. |
| `ma_slope` | `ma_slope(df: 'pd.DataFrame', period: 'int' = 50, lookback: 'int' = 10)` | Sign of the SMA slope over ``lookback`` bars. |
| `macd_momentum` | `macd_momentum(df: 'pd.DataFrame', fast: 'int' = 12, slow: 'int' = 26, signal: 'int' = 9, threshold: 'float' = 0.0)` | MACD histogram momentum with a dead zone around zero. |
| `macd_trend` | `macd_trend(df: 'pd.DataFrame', fast: 'int' = 12, slow: 'int' = 26, signal: 'int' = 9)` | Long when the MACD histogram is positive, short when negative. |
| `mfi_reversal` | `mfi_reversal(df: 'pd.DataFrame', period: 'int' = 14, lower: 'float' = 20.0, upper: 'float' = 80.0, exit_mid: 'float' = 50.0)` | Money Flow Index reversal around 20/80. |
| `momentum_3m` | `momentum_3m(df: 'pd.DataFrame', lookback: 'int' = 63)` | Sign of the 3-month (default) price momentum. |
| `momentum_acceleration` | `momentum_acceleration(df: 'pd.DataFrame', short: 'int' = 21, long: 'int' = 63)` | Sign of the difference between short and long momentum. |
| `momentum_signal` | `momentum_signal(df: 'pd.DataFrame', lookback: 'int' = 126, skip: 'int' = 21)` | 12-1 style momentum: sign of the return over ``lookback`` skipping ``skip``. |
| `momentum_vol_filter` | `momentum_vol_filter(df: 'pd.DataFrame', lookback: 'int' = 63, vol_window: 'int' = 21, vol_cap: 'float' = 0.03)` | Momentum only when recent realized volatility is below ``vol_cap``. |
| `multi_timeframe_trend` | `multi_timeframe_trend(df: 'pd.DataFrame', short: 'int' = 20, long: 'int' = 100)` | Long/short only when the short and long trend filters agree. |
| `obv_trend` | `obv_trend(df: 'pd.DataFrame', period: 'int' = 21, lookback: 'int' = 5)` | Sign of the smoothed On-Balance Volume slope. |
| `percent_b_reversion` | `percent_b_reversion(df: 'pd.DataFrame', period: 'int' = 20, mult: 'float' = 2.0, deadzone: 'float' = 0.1)` | Trade against Bollinger %B extremes with a dead zone around 0.5. |
| `pivot_structure_trade` | `pivot_structure_trade(df: 'pd.DataFrame', pct: 'float' = 0.05)` | Ride break-of-structure events until the structure flips. |
| `price_vs_sma` | `price_vs_sma(df: 'pd.DataFrame', period: 'int' = 200)` | Long above the SMA, short below it. |
| `range_fade` | `range_fade(df: 'pd.DataFrame', period: 'int' = 20, entry: 'float' = 0.9, exit_level: 'float' = 0.5)` | Fade the extremes of the rolling high-low range. |
| `regime_adaptive` | `regime_adaptive(df: 'pd.DataFrame', efficiency_period: 'int' = 20, threshold: 'float' = 0.35)` | Switch between trend following and mean reversion by the efficiency ratio. |
| `relative_volume_trend` | `relative_volume_trend(df: 'pd.DataFrame', period: 'int' = 20, threshold: 'float' = 1.2)` | Price-vs-SMA direction, gated to days with elevated relative volume. |
| `risk_adjusted_momentum_strategy` | `risk_adjusted_momentum_strategy(df: 'pd.DataFrame', lookback: 'int' = 63)` | Sign of momentum divided by realized volatility. |
| `rsi2_reversal` | `rsi2_reversal(df: 'pd.DataFrame', period: 'int' = 2, lower: 'float' = 10.0, upper: 'float' = 90.0)` | Connors-style ultra-short RSI(2) reversal entries. |
| `rsi_reversal` | `rsi_reversal(df: 'pd.DataFrame', period: 'int' = 14, lower: 'float' = 30.0, upper: 'float' = 70.0, exit_mid: 'float' = 50.0)` | Long oversold RSI, short overbought; exit at the midline. |
| `sma_cross` | `sma_cross(df: 'pd.DataFrame', fast: 'int' = 20, slow: 'int' = 50)` | Long when the fast SMA is above the slow SMA, short otherwise. |
| `squeeze_release` | `squeeze_release(df: 'pd.DataFrame')` | TTM squeeze: take the momentum side only after the squeeze releases. |
| `stoch_reversal` | `stoch_reversal(df: 'pd.DataFrame', k_period: 'int' = 14, lower: 'float' = 20.0, upper: 'float' = 80.0, exit_mid: 'float' = 50.0)` | Stochastic oscillator reversal around 20/80. |
| `stoch_rsi_reversal` | `stoch_rsi_reversal(df: 'pd.DataFrame', period: 'int' = 14, lower: 'float' = 20.0, upper: 'float' = 80.0)` | Stochastic-RSI reversal entries. |
| `supertrend_strategy` | `supertrend_strategy(df: 'pd.DataFrame', period: 'int' = 10, mult: 'float' = 3.0)` | Follow the Supertrend direction flip. |
| `support_resistance_trade` | `support_resistance_trade(df: 'pd.DataFrame', period: 'int' = 20, tol: 'float' = 0.02)` | Buy near rolling support and sell near rolling resistance. |
| `tema_trend` | `tema_trend(df: 'pd.DataFrame', period: 'int' = 20, lookback: 'int' = 5)` | Sign of the TEMA slope. |
| `trend_quality_filter` | `trend_quality_filter(df: 'pd.DataFrame', period: 'int' = 20, min_score: 'float' = 0.3)` | Trade the trend only when the trend-quality score is decisive. |
| `triple_ma` | `triple_ma(df: 'pd.DataFrame', fast: 'int' = 10, mid: 'int' = 30, slow: 'int' = 60)` | Long when fast > mid > slow, short when fast < mid < slow, else flat. |
| `trix_signal` | `trix_signal(df: 'pd.DataFrame', period: 'int' = 15)` | Sign of the TRIX rate of change. |
| `turtle_system` | `turtle_system(df: 'pd.DataFrame', entry: 'int' = 20, exit: 'int' = 10)` | Turtle-style breakout entries with shorter opposite exit channels. |
| `vol_breakout_momentum` | `vol_breakout_momentum(df: 'pd.DataFrame', period: 'int' = 20, expansion: 'float' = 1.2)` | Follow price velocity when volatility expands above its recent norm. |
| `vol_regime_switch` | `vol_regime_switch(df: 'pd.DataFrame', fast: 'int' = 21, slow: 'int' = 63)` | Trend-follow in calm regimes, mean-revert in turbulent regimes. |
| `vol_target_trend` | `vol_target_trend(df: 'pd.DataFrame', period: 'int' = 50, target_vol: 'float' = 0.2, vol_window: 'int' = 21)` | Trend positions scaled to a volatility target (returns weights in [-1, 1]). |
| `volume_breakout` | `volume_breakout(df: 'pd.DataFrame', period: 'int' = 20, vol_z: 'float' = 1.0)` | Breakout with volume confirmation; exit on the SMA cross. |
| `vortex_trend` | `vortex_trend(df: 'pd.DataFrame', period: 'int' = 14)` | Vortex trend: long when +VI leads -VI, short otherwise. |
| `vwap_cross` | `vwap_cross(df: 'pd.DataFrame', window: 'int' = 20)` | Long above the rolling VWAP, short below it. |
| `vwap_reversion` | `vwap_reversion(df: 'pd.DataFrame', window: 'int' = 20, band: 'float' = 0.02)` | Fade deviations from a rolling volume-weighted average price. |
| `williams_reversal` | `williams_reversal(df: 'pd.DataFrame', period: 'int' = 14, lower: 'float' = -80.0, upper: 'float' = -20.0, exit_mid: 'float' = -50.0)` | Williams %R reversal at overbought/oversold extremes. |
| `zscore_reversion` | `zscore_reversion(df: 'pd.DataFrame', period: 'int' = 20, entry: 'float' = 2.0, exit_level: 'float' = 0.5)` | Enter on extreme price z-scores, exit back toward the mean. |
