# New Features

Auto-generated from `quant.catalog` - do not edit by hand.
Regenerate with `python scripts/generate_quant_docs.py`.

## Overview

| Module | Kind | Items |
| --- | --- | --- |
| `quant.indicators_ext` | indicator | 137 |
| `quant.ehlers` | indicator | 24 |
| `quant.candlesticks` | candle | 90 |
| `quant.patterns_ext` | chart_pattern | 62 |
| `quant.statistics_ext` | statistic | 87 |
| `quant.features_ext` | feature | 66 |
| `quant.metrics_ext` | metric | 81 |
| `quant.risk_ext` | risk | 66 |
| `quant.portfolio_ext` | portfolio | 50 |
| `quant.strategies_ext` | strategy | 66 |
| `quant.options` | option | 63 |
| `quant.execution_ext` | execution | 54 |
| `quant.screener_ext` | scanner | 44 |
| `quant.catalog` | registry | 947 |

## Module details

### `quant.indicators_ext`

137 additional vectorized technical indicators (oscillators, trend, volume, volatility, statistics).

See [INDICATORS.md](INDICATORS.md) for the full indicator tables.

### `quant.ehlers`

Ehlers-style cycle and signal-processing filters (sine wave, dominant cycle, roofing filter, adaptive filters).

See [INDICATORS.md](INDICATORS.md) for the full indicator tables.

### `quant.candlesticks`

90 candlestick pattern detectors plus counts and a normalized pattern score.

| Name | Signature | Description |
| --- | --- | --- |
| `abandoned_baby_bear` | `abandoned_baby_bear(open_, high, low, close)` | Bearish abandoned baby: up bar, gapped-up doji, gapped-down down bar. |
| `abandoned_baby_bull` | `abandoned_baby_bull(open_, high, low, close)` | Bullish abandoned baby: down bar, gapped-down doji, gapped-up up bar. |
| `advance_block` | `advance_block(open_, high, low, close)` | Advance block: three rising bars whose bodies shrink (weakening advance). |
| `bearish_counterattack` | `bearish_counterattack(open_, high, low, close, tol_pct: 'float' = 0.1)` | Bearish counterattack: gap up then close back at the prior close. |
| `bearish_engulfing` | `bearish_engulfing(open_, high, low, close)` | Bearish engulfing: down body fully engulfs the prior up body after an advance. |
| `bearish_harami` | `bearish_harami(open_, high, low, close)` | Bearish harami: small down body contained in the prior up body after an advance. |
| `bearish_harami_cross` | `bearish_harami_cross(open_, high, low, close)` | Bearish harami cross: bearish harami whose second bar is a doji. |
| `belt_hold_bear` | `belt_hold_bear(open_, high, low, close, body_pct: 'float' = 0.7, shadow_pct: 'float' = 0.05)` | Bearish belt hold: opens at the high and runs lower. |
| `belt_hold_bull` | `belt_hold_bull(open_, high, low, close, body_pct: 'float' = 0.7, shadow_pct: 'float' = 0.05)` | Bullish belt hold: opens at the low and runs higher. |
| `black_marubozu` | `black_marubozu(open_, high, low, close, body_pct: 'float' = 0.9, shadow_pct: 'float' = 0.05)` | Black marubozu: full-length bearish body with no shadows. |
| `breakaway_bear` | `breakaway_bear(open_, high, low, close)` | Bearish breakaway: gap-up exhaustion relieved by a strong reversing bar. |
| `breakaway_bull` | `breakaway_bull(open_, high, low, close)` | Bullish breakaway: gap-down exhaustion relieved by a strong reversing bar. |
| `bullish_counterattack` | `bullish_counterattack(open_, high, low, close, tol_pct: 'float' = 0.1)` | Bullish counterattack: gap down then close back at the prior close. |
| `bullish_engulfing` | `bullish_engulfing(open_, high, low, close)` | Bullish engulfing: up body fully engulfs the prior down body after a decline. |
| `bullish_harami` | `bullish_harami(open_, high, low, close)` | Bullish harami: small up body contained in the prior down body after a decline. |
| `bullish_harami_cross` | `bullish_harami_cross(open_, high, low, close)` | Bullish harami cross: bullish harami whose second bar is a doji. |
| `closing_marubozu_black` | `closing_marubozu_black(open_, high, low, close, body_pct: 'float' = 0.85, shadow_pct: 'float' = 0.05)` | Closing marubozu (black): bearish bar closing at the low (no lower shadow). |
| `closing_marubozu_white` | `closing_marubozu_white(open_, high, low, close, body_pct: 'float' = 0.85, shadow_pct: 'float' = 0.05)` | Closing marubozu (white): bullish bar closing at the high (no upper shadow). |
| `concealing_baby_swallow` | `concealing_baby_swallow(open_, high, low, close)` | Concealing baby swallow: two down bars, an engulfing down bar, then a hammer. |
| `dark_cloud_cover` | `dark_cloud_cover(open_, high, low, close)` | Dark cloud cover: up gap open then a close below the midpoint of the prior body. |
| `deliberation` | `deliberation(open_, high, low, close)` | Deliberation: two long up bars followed by a tiny-bodied star. |
| `doji` | `doji(open_, high, low, close, body_pct: 'float' = 0.1)` | Doji: open and close nearly equal (body <= 10% of range). |
| `down_bar` | `down_bar(open_, high, low, close)` | Down bar: close below open. |
| `downside_gap_three_methods` | `downside_gap_three_methods(open_, high, low, close)` | Downside gap three methods: two falling gapped bars, third pulls back into the gap. |
| `dragonfly_doji` | `dragonfly_doji(open_, high, low, close, body_pct: 'float' = 0.1, lower_pct: 'float' = 0.6)` | Dragonfly doji: doji opening/closing at the high with a long lower shadow. |
| `evening_doji_star` | `evening_doji_star(open_, high, low, close)` | Evening doji star: evening star whose middle bar is a doji. |
| `evening_star` | `evening_star(open_, high, low, close, long_pct: 'float' = 0.5, small_pct: 'float' = 0.5)` | Evening star: long up bar, small gapped-up star, strong down bar. |
| `falling_three_methods` | `falling_three_methods(open_, high, low, close)` | Falling three methods: long down bar, three contained bounce bars, breakdown bar. |
| `four_price_doji` | `four_price_doji(open_, high, low, close, tol_pct: 'float' = 1e-06)` | Four-price doji: open, high, low, and close all equal. |
| `gravestone_doji` | `gravestone_doji(open_, high, low, close, body_pct: 'float' = 0.1, upper_pct: 'float' = 0.6)` | Gravestone doji: doji at the low with a long upper shadow. |
| `hammer` | `hammer(open_, high, low, close, shadow_mult: 'float' = 2.0)` | Hammer: small body on top of a long lower shadow after a decline. |
| `hanging_man` | `hanging_man(open_, high, low, close, shadow_mult: 'float' = 2.0)` | Hanging man: hammer shape appearing after an advance. |
| `high_wave` | `high_wave(open_, high, low, close, body_pct: 'float' = 0.2, range_mult: 'float' = 1.5, period: 'int' = 20)` | High-wave bar: unusually wide range with a small body and long shadows. |
| `identical_three_crows` | `identical_three_crows(open_, high, low, close, tol_pct: 'float' = 0.05)` | Identical three crows: three down bars each opening at the prior close. |
| `in_neck` | `in_neck(open_, high, low, close, tol_pct: 'float' = 0.1)` | In-neck: up bar closing just barely above the prior bearish close. |
| `inside_bar` | `inside_bar(open_, high, low, close)` | Inside bar: range contained within the previous bar's range. |
| `inverted_hammer` | `inverted_hammer(open_, high, low, close, shadow_mult: 'float' = 2.0)` | Inverted hammer: small body above a long upper shadow after a decline. |
| `kicking_bear` | `kicking_bear(open_, high, low, close)` | Bearish kicking: white marubozu followed by a gapped black marubozu. |
| `kicking_bull` | `kicking_bull(open_, high, low, close)` | Bullish kicking: black marubozu followed by a gapped white marubozu. |
| `ladder_bottom` | `ladder_bottom(open_, high, low, close)` | Ladder bottom: three stepped-down bars with small upper shadows, then a gap up. |
| `ladder_top` | `ladder_top(open_, high, low, close)` | Ladder top: three stepped-up bars with small lower shadows, then a gap down. |
| `long_legged_doji` | `long_legged_doji(open_, high, low, close, body_pct: 'float' = 0.1, shadow_pct: 'float' = 0.3)` | Long-legged doji: doji with long upper and lower shadows. |
| `long_lower_shadow` | `long_lower_shadow(open_, high, low, close, lower_pct: 'float' = 0.6, upper_pct: 'float' = 0.15)` | Bullish pin bar: dominant lower shadow, negligible upper shadow. |
| `long_upper_shadow` | `long_upper_shadow(open_, high, low, close, upper_pct: 'float' = 0.6, lower_pct: 'float' = 0.15)` | Bearish pin bar: dominant upper shadow, negligible lower shadow. |
| `mat_hold_bear` | `mat_hold_bear(open_, high, low, close)` | Bearish mat hold: falling three methods with a gap down on the second bar. |
| `mat_hold_bull` | `mat_hold_bull(open_, high, low, close)` | Bullish mat hold: rising three methods with a gap up on the second bar. |
| `matching_high` | `matching_high(open_, high, low, close, tol_pct: 'float' = 0.05)` | Matching high: two consecutive bullish closes at the same level after an advance. |
| `matching_low` | `matching_low(open_, high, low, close, tol_pct: 'float' = 0.05)` | Matching low: two consecutive bearish closes at the same level after a decline. |
| `meeting_lines_bear` | `meeting_lines_bear(open_, high, low, close, tol_pct: 'float' = 0.05)` | Bearish meeting lines: long up bar then a long down bar closing at the same price. |
| `meeting_lines_bull` | `meeting_lines_bull(open_, high, low, close, tol_pct: 'float' = 0.05)` | Bullish meeting lines: long down bar then a long up bar closing at the same price. |
| `morning_doji_star` | `morning_doji_star(open_, high, low, close)` | Morning doji star: morning star whose middle bar is a doji. |
| `morning_star` | `morning_star(open_, high, low, close, long_pct: 'float' = 0.5, small_pct: 'float' = 0.5)` | Morning star: long down bar, small gapped-down star, strong up bar. |
| `narrow_range_bar` | `narrow_range_bar(open_, high, low, close, period: 'int' = 20, mult: 'float' = 0.5)` | Narrow range bar: true range at most ``mult`` times its recent average. |
| `on_neck` | `on_neck(open_, high, low, close, tol_pct: 'float' = 0.05)` | On-neck: up bar closing at (or marginally below) the prior bearish close. |
| `outside_bar` | `outside_bar(open_, high, low, close)` | Outside bar: range engulfs the previous bar's range. |
| `piercing_line` | `piercing_line(open_, high, low, close)` | Piercing line: down gap open then a close above the midpoint of the prior body. |
| `rickshaw_man` | `rickshaw_man(open_, high, low, close, body_pct: 'float' = 0.1)` | Rickshaw man: long-legged doji with the tiny body centered in the range. |
| `rising_three_methods` | `rising_three_methods(open_, high, low, close)` | Rising three methods: long up bar, three contained pullback bars, breakout bar. |
| `separating_lines_bear` | `separating_lines_bear(open_, high, low, close, tol_pct: 'float' = 0.05)` | Bearish separating lines: up bar then a down bar opening at the same level. |
| `separating_lines_bull` | `separating_lines_bull(open_, high, low, close, tol_pct: 'float' = 0.05)` | Bullish separating lines: down bar then an up bar opening at the same level. |
| `shaven_bottom` | `shaven_bottom(open_, high, low, close, shadow_pct: 'float' = 0.05)` | Shaven bottom: open equals the bar low (no lower shadow). |
| `shaven_head` | `shaven_head(open_, high, low, close, shadow_pct: 'float' = 0.05)` | Shaven head: close equals the bar high (no upper shadow). |
| `shooting_star` | `shooting_star(open_, high, low, close, shadow_mult: 'float' = 2.0)` | Shooting star: inverted-hammer shape appearing after an advance. |
| `side_by_side_white_gapdown` | `side_by_side_white_gapdown(open_, high, low, close)` | Downside gap side-by-side white lines: gapped-down pair of near-identical up bars. |
| `side_by_side_white_gapup` | `side_by_side_white_gapup(open_, high, low, close)` | Upside gap side-by-side white lines: gapped-up pair of near-identical up bars. |
| `spinning_top` | `spinning_top(open_, high, low, close, max_body: 'float' = 0.3)` | Spinning top: small body with shadows on both sides. |
| `stick_sandwich` | `stick_sandwich(open_, high, low, close)` | Stick sandwich: two identical bearish closes around an up bar. |
| `takuri` | `takuri(open_, high, low, close, lower_pct: 'float' = 0.6, body_pct: 'float' = 0.2)` | Takuri: dragonfly variant with a small body at the top of the range. |
| `tasuki_gap_down` | `tasuki_gap_down(open_, high, low, close)` | Downside tasuki gap: gap down held partially unfilled by the third bar. |
| `tasuki_gap_up` | `tasuki_gap_up(open_, high, low, close)` | Upside tasuki gap: gap up held partially unfilled by the third bar. |
| `three_black_crows` | `three_black_crows(open_, high, low, close, shadow_pct: 'float' = 0.3)` | Three black crows: three falling bearish bars with small lower shadows. |
| `three_drives_bear` | `three_drives_bear(open_, high, low, close)` | Bearish three drives (heuristic): three stair-stepped advances reversing down. |
| `three_drives_bull` | `three_drives_bull(open_, high, low, close)` | Bullish three drives (heuristic): three stair-stepped declines reversing up. |
| `three_inside_down` | `three_inside_down(open_, high, low, close)` | Three inside down: bearish harami followed by a confirming down bar. |
| `three_inside_up` | `three_inside_up(open_, high, low, close)` | Three inside up: bullish harami followed by a confirming up bar. |
| `three_line_strike_bear` | `three_line_strike_bear(open_, high, low, close)` | Bearish three-line strike: three rising bars erased by one long down bar. |
| `three_line_strike_bull` | `three_line_strike_bull(open_, high, low, close)` | Bullish three-line strike: three falling bars erased by one long up bar. |
| `three_outside_down` | `three_outside_down(open_, high, low, close)` | Three outside down: bearish engulfing followed by a lower close. |
| `three_outside_up` | `three_outside_up(open_, high, low, close)` | Three outside up: bullish engulfing followed by a higher close. |
| `three_stars_in_the_south` | `three_stars_in_the_south(open_, high, low, close)` | Three stars in the south: three declining dojis, the middle one gapping lower. |
| `three_white_soldiers` | `three_white_soldiers(open_, high, low, close, shadow_pct: 'float' = 0.3)` | Three white soldiers: three rising bullish bars with small upper shadows. |
| `thrusting` | `thrusting(open_, high, low, close)` | Thrusting: down bar, gap-down up bar closing into the prior body below its midpoint. |
| `tri_star_bear` | `tri_star_bear(open_, high, low, close)` | Bearish tri-star: three dojis with the middle star gapping higher. |
| `tri_star_bull` | `tri_star_bull(open_, high, low, close)` | Bullish tri-star: three dojis with the middle star gapping lower. |
| `two_crows_gapping` | `two_crows_gapping(open_, high, low, close)` | Two crows gapping: white bar, gapped-up black bar, then a lower black bar. |
| `unique_three_river` | `unique_three_river(open_, high, low, close)` | Unique three river: down bar, lower-low small down bar, recovering up bar. |
| `up_bar` | `up_bar(open_, high, low, close)` | Up bar: close above open. |
| `upside_gap_three_methods` | `upside_gap_three_methods(open_, high, low, close)` | Upside gap three methods: two advancing gapped bars, third pulls back into the gap. |
| `white_marubozu` | `white_marubozu(open_, high, low, close, body_pct: 'float' = 0.9, shadow_pct: 'float' = 0.05)` | White marubozu: full-length bullish body with no shadows. |
| `wide_range_bar` | `wide_range_bar(open_, high, low, close, period: 'int' = 20, mult: 'float' = 2.0)` | Wide range bar: true range at least ``mult`` times its recent average. |

### `quant.patterns_ext`

Chart structure and pattern detection: pivots, break-of-structure, formations, divergences.

| Name | Signature | Description |
| --- | --- | --- |
| `ascending_channel` | `ascending_channel(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` | Ascending parallel channel. |
| `ascending_triangle` | `ascending_triangle(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, flat: 'float' = 0.02, min_slope: 'float' = 0.015, pct: 'float' = 0.05)` | Ascending triangle: flat resistance with rising support pivots. |
| `bear_flag` | `bear_flag(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pole_window: 'int' = 20, flag_window: 'int' = 10, pole_min: 'float' = 0.05)` | Bear flag: sharp decline followed by a shallow consolidation. |
| `bearish_divergence` | `bearish_divergence(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` | True where a regular bearish price/indicator divergence confirms. |
| `break_of_structure` | `break_of_structure(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | Break-of-structure events: +1 close above last swing high, -1 below last swing low. |
| `breakdown` | `breakdown(close: 'pd.Series', low: 'pd.Series', period: 'int' = 20, confirm_pct: 'float' = 0.0)` | True when close breaks below the prior rolling support by ``confirm_pct``. |
| `breakout_up` | `breakout_up(close: 'pd.Series', high: 'pd.Series', period: 'int' = 20, confirm_pct: 'float' = 0.0)` | True when close exceeds the prior rolling resistance by ``confirm_pct``. |
| `broadening_formation` | `broadening_formation(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` | Broadening formation: rising highs with falling lows (diverging). |
| `bull_flag` | `bull_flag(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pole_window: 'int' = 20, flag_window: 'int' = 10, pole_min: 'float' = 0.05)` | Bull flag: sharp advance followed by a shallow consolidation. |
| `bullish_divergence` | `bullish_divergence(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` | True where a regular bullish price/indicator divergence confirms. |
| `change_of_character` | `change_of_character(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | Change-of-character: +1 first higher low in a downtrend, -1 first lower high in an uptrend. |
| `chart_state` | `chart_state(df: 'pd.DataFrame', window: 'int' = 20, pct: 'float' = 0.05)` | One-frame market-structure summary: trend, levels, breaks, swings. |
| `cup_and_handle` | `cup_and_handle(close: 'pd.Series', window: 'int' = 80, tol: 'float' = 0.05, handle_frac: 'float' = 0.25, min_depth: 'float' = 0.06)` | Cup-and-handle: rounded base with rims within ``tol`` and a shallow handle. |
| `descending_channel` | `descending_channel(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` | Descending parallel channel. |
| `descending_triangle` | `descending_triangle(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, flat: 'float' = 0.02, min_slope: 'float' = 0.015, pct: 'float' = 0.05)` | Descending triangle: flat support with falling resistance pivots. |
| `detect_patterns` | `detect_patterns(df: 'pd.DataFrame', window: 'int' = 60, tol: 'float' = 0.03, pct: 'float' = 0.05)` | Run every chart-formation detector on an OHLCV frame. |
| `divergence_series` | `divergence_series(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100, mode: 'str' = 'regular')` | Divergence state: +1 bullish, -1 bearish, 0 none (marked at confirmation bar). |
| `double_bottom` | `double_bottom(high: 'pd.Series', low: 'pd.Series', tol: 'float' = 0.03, pct: 'float' = 0.05, min_gap: 'int' = 5)` | Double-bottom: two swing lows within ``tol`` with a peak between them. |
| `double_top` | `double_top(high: 'pd.Series', low: 'pd.Series', tol: 'float' = 0.03, pct: 'float' = 0.05, min_gap: 'int' = 5)` | Double-top: two swing highs within ``tol`` with a trough between them. |
| `falling_wedge` | `falling_wedge(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` | Falling wedge: converging downward lines (bullish bias). |
| `false_breakdown` | `false_breakdown(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 20, tol: 'float' = 0.005)` | False downside breaks: low pierces support but close recovers inside. |
| `false_breakout` | `false_breakout(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 20, tol: 'float' = 0.005)` | False upside breakouts: high pierces resistance but close falls back inside. |
| `head_and_shoulders` | `head_and_shoulders(high: 'pd.Series', low: 'pd.Series', shoulder_min: 'float' = 0.02, tol: 'float' = 0.05, pct: 'float' = 0.05)` | Head-and-shoulders top: higher middle peak with matching shoulders. |
| `hidden_bearish_divergence` | `hidden_bearish_divergence(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` | True where a hidden bearish divergence confirms (trend continuation). |
| `hidden_bullish_divergence` | `hidden_bullish_divergence(close: 'pd.Series', indicator: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` | True where a hidden bullish divergence confirms (trend continuation). |
| `higher_highs` | `higher_highs(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | True at swing highs that exceed the previous swing high. |
| `higher_lows` | `higher_lows(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | True at swing lows that hold above the previous swing low. |
| `horizontal_channel` | `horizontal_channel(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, flat: 'float' = 0.02, pct: 'float' = 0.05)` | Horizontal range: flat highs and flat lows. |
| `inverse_head_and_shoulders` | `inverse_head_and_shoulders(high: 'pd.Series', low: 'pd.Series', shoulder_min: 'float' = 0.02, tol: 'float' = 0.05, pct: 'float' = 0.05)` | Inverse head-and-shoulders: lower middle trough with matching shoulders. |
| `level_touches` | `level_touches(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 60, tol: 'float' = 0.01)` | Rolling count of bars touching (within ``tol``) either range boundary. |
| `lower_highs` | `lower_highs(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | True at swing highs that fail below the previous swing high. |
| `lower_lows` | `lower_lows(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | True at swing lows that undercut the previous swing low. |
| `macd_divergence` | `macd_divergence(close: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` | Divergence between price and the MACD histogram. |
| `obv_divergence` | `obv_divergence(close: 'pd.Series', volume: 'pd.Series', left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` | Divergence between price and On-Balance Volume. |
| `pattern_frequency` | `pattern_frequency(detections: 'pd.DataFrame')` | Count firings of each chart formation, sorted descending. |
| `pivot_high_values` | `pivot_high_values(high: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` | Pivot-high prices, NaN elsewhere. |
| `pivot_highs` | `pivot_highs(high: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` | Boolean flags where ``high`` is the maximum of a centered window. |
| `pivot_low_values` | `pivot_low_values(low: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` | Pivot-low prices, NaN elsewhere. |
| `pivot_lows` | `pivot_lows(low: 'pd.Series', left: 'int' = 2, right: 'int' = 2)` | Boolean flags where ``low`` is the minimum of a centered window. |
| `range_position` | `range_position(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 20)` | Position of close within the rolling high-low range, in [0, 1]. |
| `rectangle_pattern` | `rectangle_pattern(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 40, tol: 'float' = 0.03)` | Rectangle: tight rolling range with repeated boundary touches. |
| `resistance_distance` | `resistance_distance(close: 'pd.Series', high: 'pd.Series', period: 'int' = 20)` | Relative distance of close below the rolling resistance. |
| `rising_wedge` | `rising_wedge(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05)` | Rising wedge: converging upward lines (bearish bias). |
| `rolling_resistance` | `rolling_resistance(high: 'pd.Series', period: 'int' = 20)` | Rolling maximum high as dynamic resistance. |
| `rolling_support` | `rolling_support(low: 'pd.Series', period: 'int' = 20)` | Rolling minimum low as dynamic support. |
| `rounding_bottom` | `rounding_bottom(close: 'pd.Series', window: 'int' = 60, min_r2: 'float' = 0.5)` | Rounding bottom: concave-up quadratic fit with the trough mid-window. |
| `rounding_top` | `rounding_top(close: 'pd.Series', window: 'int' = 60, min_r2: 'float' = 0.5)` | Rounding top: concave-down quadratic fit with the peak mid-window. |
| `rsi_divergence` | `rsi_divergence(close: 'pd.Series', period: 'int' = 14, left: 'int' = 3, right: 'int' = 3, lookback: 'int' = 100)` | Divergence between price and RSI. |
| `structure_score` | `structure_score(detections: 'pd.DataFrame', window: 'int' = 5)` | Net formation score in [-1, 1]: bullish minus bearish firings, smoothed. |
| `support_distance` | `support_distance(close: 'pd.Series', low: 'pd.Series', period: 'int' = 20)` | Relative distance of close above the rolling support. |
| `swing_count` | `swing_count(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, pct: 'float' = 0.05, kind: 'str' = 'both')` | Rolling count of zigzag pivots within ``window`` bars. |
| `swing_high_distance` | `swing_high_distance(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | Relative distance from close to the latest swing-high level. |
| `swing_levels` | `swing_levels(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | Forward-filled latest swing-high and swing-low levels per bar. |
| `swing_low_distance` | `swing_low_distance(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | Relative distance from close to the latest swing-low level. |
| `swing_points` | `swing_points(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | Zigzag swing pivots as a frame with ``bar``, ``price``, ``kind``. |
| `swing_series` | `swing_series(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | Step-wise swing line: price at the latest confirmed pivot, forward-filled. |
| `symmetric_triangle` | `symmetric_triangle(high: 'pd.Series', low: 'pd.Series', window: 'int' = 60, flat: 'float' = 0.02, min_slope: 'float' = 0.015, pct: 'float' = 0.05)` | Symmetric triangle: falling highs into rising lows (converging). |
| `trend_structure` | `trend_structure(high: 'pd.Series', low: 'pd.Series', pct: 'float' = 0.05)` | Market structure state: +1 uptrend, -1 downtrend, 0 mixed/undefined. |
| `triple_bottom` | `triple_bottom(high: 'pd.Series', low: 'pd.Series', tol: 'float' = 0.03, pct: 'float' = 0.05, min_gap: 'int' = 5)` | Triple-bottom: three swing lows within ``tol``. |
| `triple_top` | `triple_top(high: 'pd.Series', low: 'pd.Series', tol: 'float' = 0.03, pct: 'float' = 0.05, min_gap: 'int' = 5)` | Triple-top: three swing highs within ``tol``. |
| `v_bottom` | `v_bottom(close: 'pd.Series', window: 'int' = 30, min_move: 'float' = 0.03)` | V-bottom: sharp trough with both legs moving at least ``min_move``. |
| `v_top` | `v_top(close: 'pd.Series', window: 'int' = 30, min_move: 'float' = 0.03)` | V-top: sharp peak with both legs falling at least ``min_move``. |

### `quant.statistics_ext`

Statistical toolkit: covariance estimulators, PCA, KMeans, GMM, Kalman, GARCH/OU, entropy, bootstrap.

| Name | Signature | Description |
| --- | --- | --- |
| `GaussianMixture1D` | `GaussianMixture1D(n_components: 'int' = 2, iters: 'int' = 200, tol: 'float' = 1e-06, seed: 'int' = 42)` | Univariate Gaussian mixture fitted by expectation-maximization. |
| `KMeans` | `KMeans(n_clusters: 'int' = 3, iters: 'int' = 100, seed: 'int' = 42)` | K-means clustering with k-means++ style seeding (numpy-only). |
| `KalmanFilter1D` | `KalmanFilter1D(process_var: 'float' = 1e-05, measurement_var: 'float' = 0.001, initial: 'float | None' = None)` | Random-walk-plus-noise Kalman filter for 1-D series. |
| `KalmanHedgeRatio` | `KalmanHedgeRatio(process_var: 'float' = 1e-05, measurement_var: 'float' = 0.001, initial_beta: 'float' = 1.0)` | Time-varying hedge ratio between two series via a Kalman filter. |
| `LogisticRegressionLite` | `LogisticRegressionLite(lr: 'float' = 0.1, epochs: 'int' = 500, l2: 'float' = 0.0001, standardize: 'bool' = True)` | Binary logistic regression trained with full-batch gradient descent. |
| `PCA` | `PCA(n_components: 'int' = 2)` | Principal component analysis via SVD (numpy-only). |
| `adf_test` | `adf_test(series, max_lag: 'int' = 1)` | Augmented Dickey-Fuller test statistic (constant, no trend). |
| `approximate_entropy` | `approximate_entropy(series, m: 'int' = 2, r: 'float | None' = None)` | Approximate entropy (Pincus). |
| `ar_fit` | `ar_fit(series, order: 'int' = 1)` | Autoregressive model fit via the Yule-Walker equations. |
| `ar_predict` | `ar_predict(series, coef, steps: 'int' = 1, intercept: 'float' = 0.0)` | Multi-step forecast with an AR model (coef ordered most-recent first). |
| `autocorr` | `autocorr(series, lag: 'int' = 1)` | Autocorrelation at ``lag``. |
| `binary_segmentation_changepoints` | `binary_segmentation_changepoints(series, max_changepoints: 'int' = 5, min_size: 'int' = 10)` | Binary segmentation change-point detection (variance criterion). |
| `biquad_highpass` | `biquad_highpass(series: 'pd.Series', cutoff: 'float' = 0.1, q: 'float' = 0.707)` | Second-order high-pass filter (RBJ biquad); ``cutoff`` in (0, 1). |
| `biquad_lowpass` | `biquad_lowpass(series: 'pd.Series', cutoff: 'float' = 0.1, q: 'float' = 0.707)` | Second-order Butterworth-style low-pass filter (RBJ biquad). |
| `block_bootstrap` | `block_bootstrap(x, block: 'int' = 10, n: 'int | None' = None, seed: 'int' = 42)` | Moving-block bootstrap preserving short-range dependence. |
| `bootstrap_ci` | `bootstrap_ci(x, stat_fn=None, n_boot: 'int' = 1000, alpha: 'float' = 0.05, seed: 'int' = 42)` | Non-parametric bootstrap confidence interval for a statistic. |
| `chi2_sf` | `chi2_sf(x: 'float', df: 'int')` | Chi-square survival function via the Wilson-Hilferty approximation. |
| `clip_outliers` | `clip_outliers(series: 'pd.Series', n_std: 'float' = 3.0)` | Clip values beyond ``n_std`` standard deviations from the rolling mean. |
| `covariance` | `covariance(x, y)` | Sample covariance. |
| `cross_autocorr` | `cross_autocorr(x, y, lag: 'int' = 1)` | Cross-correlation between x and lagged y. |
| `cusum_changepoints` | `cusum_changepoints(series, threshold: 'float' = 5.0, drift: 'float' = 0.5)` | CUSUM change-point detection on a standardized series. |
| `detrend_linear` | `detrend_linear(series: 'pd.Series')` | Remove a least-squares linear trend. |
| `detrend_rolling` | `detrend_rolling(series: 'pd.Series', period: 'int' = 60)` | Remove a rolling-mean trend. |
| `diff_series` | `diff_series(series: 'pd.Series', order: 'int' = 1)` | Apply differencing ``order`` times. |
| `durbin_watson` | `durbin_watson(residuals)` | Durbin-Watson statistic (2.0 = no autocorrelation). |
| `engle_granger` | `engle_granger(y, x)` | Engle-Granger cointegration: OLS beta plus ADF test on the residuals. |
| `ewma_corr` | `ewma_corr(returns: 'pd.DataFrame', lam: 'float' = 0.94)` | Exponentially weighted correlation matrix of a returns DataFrame. |
| `ewma_cov` | `ewma_cov(returns: 'pd.DataFrame', lam: 'float' = 0.94)` | Exponentially weighted covariance matrix of a returns DataFrame. |
| `fft_dominant_cycles` | `fft_dominant_cycles(series, top_n: 'int' = 3)` | Strongest spectral cycles as (period, amplitude, power) rows. |
| `fit_laplace` | `fit_laplace(x)` | Method-of-moments Laplace fit: (location, scale). |
| `fit_lognormal` | `fit_lognormal(x)` | Method-of-moments lognormal fit on positive data: (mu, sigma) of log x. |
| `fit_normal` | `fit_normal(x)` | Method-of-moments normal fit: (mean, std). |
| `fit_t` | `fit_t(x)` | Rough Student-t fit (kurtosis matching): (df, loc, scale). |
| `fractal_dimension_higuchi` | `fractal_dimension_higuchi(series, kmax: 'int' = 10)` | Higuchi fractal dimension estimate. |
| `fractal_dimension_katz` | `fractal_dimension_katz(series)` | Katz fractal dimension estimate (uses the mean of the series length). |
| `fractal_dimension_petrosian` | `fractal_dimension_petrosian(series)` | Petrosian fractal dimension estimate from sign changes. |
| `fractional_diff` | `fractional_diff(series: 'pd.Series', d: 'float' = 0.4, threshold: 'float' = 0.0001)` | Fractional differencing with fixed-width weights (Lopez de Prado). |
| `garch11_fit` | `garch11_fit(returns)` | GARCH(1,1) fit by maximum likelihood (Nelder-Mead on the log-likelihood). |
| `gpd_fit` | `gpd_fit(excesses)` | Generalized Pareto fit via probability-weighted moments: (xi, beta). |
| `half_life` | `half_life(series)` | Mean-reversion half-life from an AR(1) fit of the series on its lag. |
| `huber_regression` | `huber_regression(X, y, c: 'float' = 1.345, iters: 'int' = 25)` | Huber robust regression via iteratively reweighted least squares. |
| `hurst_dfa` | `hurst_dfa(series, scales: 'tuple[int, ...]' = (8, 16, 32, 64, 128))` | Hurst exponent via detrended fluctuation analysis. |
| `james_stein_shrink` | `james_stein_shrink(x, target: 'float | None' = None)` | James-Stein mean shrinkage toward a common target (default 0). |
| `jarque_bera` | `jarque_bera(x)` | Jarque-Bera normality test. p-value is exact for the chi2(2) limit. |
| `kendall_tau` | `kendall_tau(x, y)` | Kendall tau-b rank correlation (O(n^2), intended for research-sized data). |
| `kpss_stat` | `kpss_stat(series, lags: 'int' = 5)` | KPSS stationarity test statistic (level-stationary null). |
| `lead_lag_corr` | `lead_lag_corr(x, y, max_lag: 'int' = 10)` | Correlation of x with y shifted over [-max_lag, max_lag]. |
| `ledoit_wolf_shrinkage` | `ledoit_wolf_shrinkage(returns: 'pd.DataFrame')` | Ledoit-Wolf covariance shrinkage toward the scaled identity target. |
| `ljung_box` | `ljung_box(series, lags: 'int' = 10)` | Ljung-Box test for serial correlation; p-value via Wilson-Hilferty chi2. |
| `mad` | `mad(x, constant: 'float' = 1.4826)` | Median absolute deviation (scaled to be normal-consistent by default). |
| `mutual_information` | `mutual_information(x, y, bins: 'int' = 10)` | Histogram-based mutual information. |
| `nelder_mead` | `nelder_mead(func, x0, maxiter: 'int' = 500, tol: 'float' = 1e-08, step: 'float' = 0.1)` | Nelder-Mead simplex minimizer. Returns ``(x_best, f_best, n_iter)``. |
| `norm_cdf` | `norm_cdf(z: 'float')` | Standard normal CDF. |
| `ols_fit` | `ols_fit(X, y)` | OLS with intercept. Accepts 1-D or 2-D ``X``. |
| `ou_process_fit` | `ou_process_fit(spread)` | Ornstein-Uhlenbeck fit via AR(1) on the spread. |
| `partial_autocorr` | `partial_autocorr(series, nlags: 'int' = 10)` | Partial autocorrelations via the Durbin-Levinson recursion. |
| `pearson` | `pearson(x, y)` | Pearson correlation coefficient. |
| `percentile_of_score` | `percentile_of_score(series: 'pd.Series', score: 'float')` | Percentile (0..100) of ``score`` within ``series``. |
| `periodogram` | `periodogram(series)` | One-sided periodogram: frequency and power columns. |
| `permutation_entropy` | `permutation_entropy(series, order: 'int' = 3, delay: 'int' = 1)` | Normalized permutation entropy in [0, 1]. |
| `permutation_test` | `permutation_test(x, y, stat_fn=None, n: 'int' = 1000, seed: 'int' = 42)` | Two-sided permutation test on a two-sample statistic (mean difference). |
| `polyfit_series` | `polyfit_series(series: 'pd.Series', degree: 'int' = 2)` | Polynomial coefficients fitting the series against time (highest first). |
| `rank_ic` | `rank_ic(pred, actual)` | Rank information coefficient (Spearman correlation). |
| `ridge_regression` | `ridge_regression(X, y, alpha: 'float' = 1.0)` | Ridge regression (closed form, intercept unpenalized). |
| `rolling_autocorr` | `rolling_autocorr(series: 'pd.Series', lag: 'int' = 1, period: 'int' = 60)` | Rolling autocorrelation at the given lag. |
| `rolling_hurst` | `rolling_hurst(series: 'pd.Series', period: 'int' = 250, min_window: 'int' = 8)` | Rolling Hurst exponent via rescaled range (R/S). |
| `rolling_jarque_bera` | `rolling_jarque_bera(series: 'pd.Series', period: 'int' = 60)` | Rolling Jarque-Bera statistic. |
| `rolling_kurt` | `rolling_kurt(series: 'pd.Series', period: 'int' = 60)` | Rolling excess kurtosis. |
| `rolling_mad` | `rolling_mad(series: 'pd.Series', period: 'int' = 20, constant: 'float' = 1.4826)` | Rolling median absolute deviation. |
| `rolling_moment` | `rolling_moment(series: 'pd.Series', order: 'int' = 3, period: 'int' = 60)` | Rolling central moment of the given order. |
| `rolling_regression` | `rolling_regression(y: 'pd.Series', x: 'pd.Series', period: 'int' = 60)` | Rolling OLS of y on x; returns beta, alpha, r2, residual std. |
| `rolling_skew` | `rolling_skew(series: 'pd.Series', period: 'int' = 60)` | Rolling skewness. |
| `rolling_spearman` | `rolling_spearman(a: 'pd.Series', b: 'pd.Series', period: 'int' = 60)` | Rolling Spearman correlation. |
| `sample_entropy` | `sample_entropy(series, m: 'int' = 2, r: 'float | None' = None)` | Sample entropy (Richman & Moorman). |
| `savgol_filter` | `savgol_filter(series: 'pd.Series', window: 'int' = 11, poly: 'int' = 3)` | Savitzky-Golay smoothing filter (local polynomial least squares). |
| `seasonal_decompose` | `seasonal_decompose(series, period: 'int' = 5)` | Classical additive decomposition into trend, seasonal, and residual. |
| `skew_kurt_ratio` | `skew_kurt_ratio(x)` | Skewness divided by excess kurtosis (tail asymmetry balance). |
| `spearman` | `spearman(x, y)` | Spearman rank correlation coefficient. |
| `spectral_entropy` | `spectral_entropy(series, bins: 'int' = 32)` | Entropy of the normalized power spectrum in [0, 1]. |
| `stationary_bootstrap` | `stationary_bootstrap(x, mean_block: 'float' = 10.0, n: 'int | None' = None, seed: 'int' = 42)` | Stationary bootstrap with geometric block lengths (Politis & Romano). |
| `transfer_entropy` | `transfer_entropy(x, y, lag: 'int' = 1, bins: 'int' = 5)` | Histogram-based transfer entropy from x to y (nats). |
| `variance_ratio` | `variance_ratio(series, q: 'int' = 2)` | Lo-MacKinlay variance ratio: VR(q) = Var(q-period returns) / (q * Var(1-period)). |
| `variance_ratio_z` | `variance_ratio_z(series, q: 'int' = 2)` | Variance ratio with the homoskedasticity-robust z-statistic. |
| `wavelet_energy` | `wavelet_energy(series, levels: 'int' = 3)` | Energy share of each Haar wavelet detail level plus the approximation. |
| `wavelet_haar` | `wavelet_haar(series, levels: 'int' = 3)` | Haar discrete wavelet decomposition. |
| `winsorize` | `winsorize(series: 'pd.Series', lower: 'float' = 0.01, upper: 'float' = 0.99)` | Clip a series at empirical quantiles. |
| `zscore_df` | `zscore_df(df: 'pd.DataFrame', period: 'int' = 60)` | Cross-time z-score of every column. |

### `quant.features_ext`

Feature engineering: returns, volatility, momentum, mean-reversion, calendar, entropy, targets, selectors, FeaturePipeline.

| Name | Signature | Description |
| --- | --- | --- |
| `FeaturePipeline` | `FeaturePipeline(impute: 'bool' = True, winsorize: 'tuple[float, float] | None' = (0.01, 0.99), scale: 'str | None' = 'zscore', corr_threshold: 'float | None' = 0.95, top_k: 'int | None' = None, ic_method: 'str' = 'spearman', n_components: 'int | None' = None)` | Leakage-safe feature transform pipeline. |
| `beta_features` | `beta_features(df: 'pd.DataFrame', benchmark_df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (60, 252))` | Rolling market beta/alpha of the asset against a benchmark. |
| `bollinger_features` | `bollinger_features(df: 'pd.DataFrame', period: 'int' = 20, num_std: 'float' = 2.0)` | Bollinger %B, width, and squeeze flags. |
| `build_feature_matrix` | `build_feature_matrix(df: 'pd.DataFrame', groups: 'tuple[str, ...] | None' = None, dropna: 'bool' = False)` | Assemble common feature groups into a single design matrix. |
| `calendar_features` | `calendar_features(index: 'pd.Index')` | Calendar/seasonality features from a DatetimeIndex. |
| `candlestick_features` | `candlestick_features(df: 'pd.DataFrame', score_window: 'int' = 5)` | Aggregated bull/bear pattern counts and smoothed net candle score. |
| `corwin_schultz_spread` | `corwin_schultz_spread(high: 'pd.Series', low: 'pd.Series', window: 'int' = 21)` | Corwin-Schultz high-low spread estimator (2-bar form, rolling mean). |
| `cross_asset_features` | `cross_asset_features(df: 'pd.DataFrame', benchmark_df: 'pd.DataFrame')` | Relative strength, rolling correlation, and spread z vs a benchmark. |
| `cycle_features` | `cycle_features(df: 'pd.DataFrame')` | Ehlers dominant cycle, sine phase, and period-normalized lead. |
| `days_to_month_end` | `days_to_month_end(index: 'pd.Index')` | Trading-day distance to the end of the calendar month. |
| `diff_features` | `diff_features(df: 'pd.DataFrame', orders: 'tuple[int, ...]' = (1, 2, 5, 21))` | N-th discrete differences of every column, suffix ``_diff{n}``. |
| `drop_correlated` | `drop_correlated(df: 'pd.DataFrame', threshold: 'float' = 0.95, method: 'str' = 'pearson')` | Greedily drop later columns correlated above ``threshold`` with an earlier one. |
| `drop_low_variance` | `drop_low_variance(df: 'pd.DataFrame', threshold: 'float' = 1e-08)` | Drop columns whose variance is below ``threshold``. |
| `encode_cyclical` | `encode_cyclical(series: 'pd.Series', period: 'float')` | Sine/cosine encoding of a cyclical integer feature. |
| `entropy_features` | `entropy_features(df: 'pd.DataFrame', window: 'int' = 60)` | Rolling permutation/approximate/spectral entropy of returns. |
| `ewma_features` | `ewma_features(df: 'pd.DataFrame', halflives: 'tuple[int, ...]' = (5, 21, 63))` | EWM means of every column for several half-lives, suffix ``_ewm{h}``. |
| `feature_summary` | `feature_summary(df: 'pd.DataFrame')` | Per-column dtype/nan/mean/std/min/max overview of a feature frame. |
| `high_52w_features` | `high_52w_features(df: 'pd.DataFrame', lookback: 'int' = 252)` | Distance to the 52-week high/low and the yearly position. |
| `impute_median_df` | `impute_median_df(df: 'pd.DataFrame')` | Fill missing values with each column median. |
| `interaction_features` | `interaction_features(df: 'pd.DataFrame', pairs: 'list[tuple[str, str]] | None' = None)` | Pairwise products and ratios; defaults to adjacent-column pairs. |
| `intraday_feature` | `intraday_feature(df: 'pd.DataFrame')` | Intraday open-to-close returns and their rolling sums. |
| `intramonth_position` | `intramonth_position(index: 'pd.Index')` | Position within the month in [0, 1] (0 = month start, 1 = month end). |
| `kyle_lambda` | `kyle_lambda(close: 'pd.Series', volume: 'pd.Series', window: 'int' = 60)` | Rolling Kyle's lambda: return sensitivity to signed volume flow. |
| `lag_features` | `lag_features(df: 'pd.DataFrame', lags: 'tuple[int, ...]' = (1, 2, 3, 5, 10))` | Lag every column at the given horizons, with suffix ``_lag{n}``. |
| `lead_lag_features` | `lead_lag_features(series_a: 'pd.Series', series_b: 'pd.Series', lags: 'tuple[int, ...]' = (-3, -2, -1, 0, 1, 2, 3), window: 'int' = 60)` | Rolling lead/lag correlations between two series. |
| `liquidity_features` | `liquidity_features(df: 'pd.DataFrame', window: 'int' = 21)` | Liquidity proxies: Amihud illiquidity, turnover z, and spread estimates. |
| `mean_reversion_features` | `mean_reversion_features(df: 'pd.DataFrame', window: 'int' = 63)` | Half-life, variance ratio, Hurst exponent, and efficiency of mean reversion. |
| `meta_labels` | `meta_labels(primary: 'pd.Series', tb_labels: 'pd.Series')` | Meta-label: 1 when the primary side agrees with the triple-barrier outcome. |
| `microstructure_features` | `microstructure_features(df: 'pd.DataFrame', window: 'int' = 21)` | Microstructure pack: rolled spread, Corwin-Schultz, Kyle lambda, noise. |
| `momentum_features` | `momentum_features(df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (5, 10, 21, 63, 126, 252))` | Return grid plus composite momentum score and efficiency ratio. |
| `momentum_quality_features` | `momentum_quality_features(df: 'pd.DataFrame', window: 'int' = 63)` | Momentum quality: hit rate, sign consistency, and reward/risk of trend. |
| `oscillator_features` | `oscillator_features(df: 'pd.DataFrame')` | RSI/MACD/Stochastic/Williams %R/CCI grid on one frame. |
| `overnight_feature` | `overnight_feature(df: 'pd.DataFrame')` | Overnight gap and its rolling sums. |
| `pca_features` | `pca_features(X: 'pd.DataFrame', n_components: 'int' = 5)` | Principal components of a numeric frame (rows with NaNs yield NaN PCs). |
| `price_level_features` | `price_level_features(df: 'pd.DataFrame')` | Typical/median/weighted price levels normalized by close. |
| `price_zscore_features` | `price_zscore_features(df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (20, 60, 252))` | Z-score of price against its rolling mean for several windows. |
| `range_features` | `range_features(df: 'pd.DataFrame')` | Bar geometry: shadows, body, close location, and gaps. |
| `rank_transform_df` | `rank_transform_df(df: 'pd.DataFrame')` | Convert every column to percentile ranks in [0, 1]. |
| `return_features` | `return_features(df: 'pd.DataFrame', horizons: 'tuple[int, ...]' = (1, 2, 5, 10, 21, 63, 126, 252), log: 'bool' = False)` | Simple (or log) returns across a grid of horizons. |
| `reversal_features` | `reversal_features(df: 'pd.DataFrame')` | Short-term reversal signals (negative of recent returns). |
| `risk_adjusted_momentum` | `risk_adjusted_momentum(df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (21, 63, 126))` | Momentum divided by realized volatility for several windows. |
| `rolling_rank_features` | `rolling_rank_features(df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (21, 63, 252))` | Percentile rank of price within its trailing window. |
| `rolling_stat_features` | `rolling_stat_features(df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (10, 20, 60), stats: 'tuple[str, ...]' = ('mean', 'std', 'skew'))` | Rolling mean/std/skew/min/max/median for every numeric column. |
| `sample_weights_by_vol` | `sample_weights_by_vol(df: 'pd.DataFrame', window: 'int' = 21)` | Inverse-volatility sample weights, normalized to mean 1. |
| `scale_minmax_df` | `scale_minmax_df(df: 'pd.DataFrame')` | Min-max scale every column into [0, 1]. |
| `scale_zscore_df` | `scale_zscore_df(df: 'pd.DataFrame')` | Z-score standardize every column. |
| `select_by_ic` | `select_by_ic(X: 'pd.DataFrame', y: 'pd.Series', top_k: 'int' = 20, method: 'str' = 'spearman')` | Select the ``top_k`` features with the largest |IC| against ``y``. |
| `select_by_mutual_info` | `select_by_mutual_info(X: 'pd.DataFrame', y: 'pd.Series', top_k: 'int' = 20, bins: 'int' = 10)` | Select the ``top_k`` features by mutual information with ``y``. |
| `semivariance_features` | `semivariance_features(df: 'pd.DataFrame', window: 'int' = 21, annualize: 'int' = 252)` | Downside and upside semivariance plus their ratio. |
| `statistical_features` | `statistical_features(df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (20, 60, 120))` | Rolling skew, kurtosis, and lag-1 autocorrelation of returns. |
| `target_binary_direction` | `target_binary_direction(df: 'pd.DataFrame', horizon: 'int' = 5, threshold: 'float' = 0.0)` | Binary label: 1 if forward return exceeds ``threshold`` else 0. |
| `target_forward_returns` | `target_forward_returns(df: 'pd.DataFrame', horizons: 'tuple[int, ...]' = (1, 5, 21))` | Forward simple returns for each horizon (columns ``fwd_ret_{h}``). |
| `target_quantile_labels` | `target_quantile_labels(df: 'pd.DataFrame', horizon: 'int' = 5, n_quantiles: 'int' = 3)` | Quantile-bucketed forward-return labels in ``0..n_quantiles-1``. |
| `target_trend_labels` | `target_trend_labels(df: 'pd.DataFrame', horizon: 'int' = 21, ma_window: 'int' = 20)` | 1 if the simple MA is higher ``horizon`` bars ahead, else 0. |
| `target_triple_barrier` | `target_triple_barrier(df: 'pd.DataFrame', horizon: 'int' = 21, pt_mult: 'float' = 2.0, sl_mult: 'float' = 1.0, vol_window: 'int' = 21)` | Triple-barrier labels (+1/-1/0) with volatility-scaled barriers. |
| `target_volatility_labels` | `target_volatility_labels(df: 'pd.DataFrame', horizon: 'int' = 21, n_quantiles: 'int' = 3)` | Quantile label of future realized volatility over ``horizon`` bars. |
| `trend_features` | `trend_features(df: 'pd.DataFrame')` | Trend state: MA distances, ADX, slope/R2 of log price, trend scores. |
| `turn_of_month` | `turn_of_month(index: 'pd.Index', window: 'int' = 3)` | Boolean flag: within ``window`` trading days of month start or end. |
| `vol_of_vol_features` | `vol_of_vol_features(df: 'pd.DataFrame', vol_window: 'int' = 21, vov_windows: 'tuple[int, ...]' = (21, 63))` | Volatility-of-volatility and volatility momentum features. |
| `vol_percentile_features` | `vol_percentile_features(df: 'pd.DataFrame', vol_window: 'int' = 21, lookback: 'int' = 252)` | Percentile rank of current volatility within its trailing history. |
| `vol_regime_features` | `vol_regime_features(df: 'pd.DataFrame', window: 'int' = 21, threshold: 'float' = 0.75)` | Coarse volatility regime flags (high when above the threshold percentile). |
| `volatility_features` | `volatility_features(df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (5, 10, 21, 63, 252), annualize: 'int' = 252)` | Realized volatility grid plus ATR percent and range-based estimators. |
| `volume_features` | `volume_features(df: 'pd.DataFrame', windows: 'tuple[int, ...]' = (5, 20, 60))` | Volume z-scores, relative volume, dollar-volume z, and volume slope. |
| `volume_flow_features` | `volume_flow_features(df: 'pd.DataFrame')` | OBV/AD/CMF/MFI/WAD trend-flow features, normalized where possible. |
| `vwap_features` | `vwap_features(df: 'pd.DataFrame', window: 'int' = 21)` | Rolling-VWAP deviation and z-score features. |
| `winsorize_df` | `winsorize_df(df: 'pd.DataFrame', lower: 'float' = 0.01, upper: 'float' = 0.99)` | Clip every column to its empirical ``[lower, upper]`` quantiles. |

### `quant.metrics_ext`

Performance metrics: risk-adjusted returns, drawdowns, VaR, trade stats, benchmark-relative, strategy_metrics.

| Name | Signature | Description |
| --- | --- | --- |
| `alpha` | `alpha(returns, benchmark, risk_free: 'float' = 0.0, periods_per_year: 'int' = 252)` | Annualized Jensen's alpha against a benchmark. |
| `annualized_return` | `annualized_return(returns, periods_per_year: 'int' = 252)` | Arithmetic mean return scaled to a year. |
| `annualized_volatility` | `annualized_volatility(returns, periods_per_year: 'int' = 252)` | Standard deviation scaled to a year. |
| `average_drawdown` | `average_drawdown(returns)` | Mean of non-zero drawdown values (negative). |
| `average_holding_period` | `average_holding_period(positions, returns=None)` | Mean bar duration of non-zero position runs. |
| `average_loss` | `average_loss(trades)` | Mean PnL of losing trades (negative). |
| `average_return` | `average_return(returns)` | Mean per-period return. |
| `average_win` | `average_win(trades)` | Mean PnL of winning trades. |
| `benchmark_r_squared` | `benchmark_r_squared(returns, benchmark)` | R-squared of the returns/benchmark regression. |
| `beta` | `beta(returns, benchmark)` | OLS beta of returns against a benchmark. |
| `cagr` | `cagr(returns, periods_per_year: 'int' = 252)` | Compound annual growth rate from per-period returns. |
| `calmar_ratio` | `calmar_ratio(returns, periods_per_year: 'int' = 252)` | CAGR divided by the absolute maximum drawdown. |
| `capture_ratio` | `capture_ratio(returns, benchmark)` | Up capture divided by down capture. |
| `common_sense_ratio` | `common_sense_ratio(returns, periods_per_year: 'int' = 252)` | Profit factor times tail ratio - a compact reward/risk composite. |
| `compare_strategies` | `compare_strategies(returns, benchmark=None, periods_per_year: 'int' = 252)` | Metrics table with one column per strategy. |
| `correlation_to_benchmark` | `correlation_to_benchmark(returns, benchmark)` | Pearson correlation of returns with the benchmark. |
| `cvar_historical` | `cvar_historical(returns, level: 'float' = 0.05)` | Historical CVaR/expected shortfall beyond the VaR threshold. |
| `cvar_parametric` | `cvar_parametric(returns, level: 'float' = 0.05)` | Gaussian parametric CVaR (expected shortfall). |
| `day_of_week_returns` | `day_of_week_returns(returns)` | Mean return by weekday (0 = Monday). |
| `down_capture` | `down_capture(returns, benchmark)` | Average down-bar return divided by the benchmark's over down bars. |
| `downside_deviation` | `downside_deviation(returns, mar: 'float' = 0.0, periods_per_year: 'int' = 252)` | Annualized downside deviation below the minimum acceptable return. |
| `downside_ratio` | `downside_ratio(returns, mar: 'float' = 0.0, periods_per_year: 'int' = 252)` | Annualized excess return over downside deviation. |
| `drawdown_at_risk` | `drawdown_at_risk(returns, quantile: 'float' = 0.05)` | Worst drawdown at the given quantile of the drawdown distribution. |
| `drawdown_duration_series` | `drawdown_duration_series(returns)` | Bars elapsed since the last equity peak, for each bar. |
| `drawdown_series` | `drawdown_series(returns)` | Drawdown series in [-1, 0] relative to the running equity peak. |
| `equity_curve` | `equity_curve(returns, initial: 'float' = 1.0)` | Compounded equity curve starting at ``initial``. |
| `excess_returns` | `excess_returns(returns, benchmark)` | Per-bar excess return of the strategy over the benchmark. |
| `expectancy` | `expectancy(trades)` | Expected PnL per trade: win rate times payoff minus loss rate. |
| `exposure` | `exposure(positions)` | Fraction of bars with a non-zero position. |
| `gain_to_pain_ratio` | `gain_to_pain_ratio(returns)` | Sum of returns divided by the absolute sum of negative returns. |
| `information_ratio` | `information_ratio(returns, benchmark, periods_per_year: 'int' = 252)` | Annualized mean excess return divided by tracking error. |
| `kelly_criterion` | `kelly_criterion(trades)` | Kelly fraction from win rate and average win/loss sizes. |
| `kupiec_pof_test` | `kupiec_pof_test(returns, level: 'float' = 0.05, window: 'int' = 250)` | Kupiec proportion-of-failures VaR backtest. |
| `largest_loss` | `largest_loss(trades)` | Largest single losing trade (negative). |
| `largest_win` | `largest_win(trades)` | Largest single winning trade. |
| `martin_ratio` | `martin_ratio(returns, periods_per_year: 'int' = 252)` | CAGR divided by the ulcer index. |
| `max_consecutive_losses` | `max_consecutive_losses(trades)` | Longest run of consecutive losing trades. |
| `max_consecutive_wins` | `max_consecutive_wins(trades)` | Longest run of consecutive winning trades. |
| `max_drawdown` | `max_drawdown(returns)` | Most negative drawdown (as a negative fraction, 0 if never below peak). |
| `max_drawdown_duration` | `max_drawdown_duration(returns)` | Longest number of bars the equity stayed below a prior peak. |
| `median_return` | `median_return(returns)` | Median per-period return. |
| `modified_var` | `modified_var(returns, level: 'float' = 0.05)` | Cornish-Fisher modified VaR accounting for skew and excess kurtosis. |
| `monthly_returns_table` | `monthly_returns_table(returns)` | Year-by-month compounded return table. |
| `omega_ratio` | `omega_ratio(returns, threshold: 'float' = 0.0)` | Omega ratio: gains above threshold over losses below it. |
| `pain_index` | `pain_index(returns)` | Mean drawdown magnitude (positive percent). |
| `payoff_ratio` | `payoff_ratio(trades)` | Average win divided by the absolute average loss. |
| `position_pnl` | `position_pnl(positions, returns)` | Per-bar strategy PnL from positions lagged one bar times returns. |
| `position_pnl_curve` | `position_pnl_curve(positions, returns, initial: 'float' = 1.0)` | Compounded equity curve of the position PnL. |
| `profit_factor` | `profit_factor(trades)` | Gross profits divided by gross losses. |
| `profit_per_trade` | `profit_per_trade(trades)` | Mean PnL per trade (alias of expectancy). |
| `recovery_factor` | `recovery_factor(returns)` | Total return divided by the absolute maximum drawdown. |
| `return_kurtosis` | `return_kurtosis(returns)` | Excess kurtosis of per-period returns. |
| `return_skew` | `return_skew(returns)` | Skewness of per-period returns. |
| `risk_return_ratio` | `risk_return_ratio(returns)` | Mean per-period return divided by per-period standard deviation. |
| `rolling_alpha` | `rolling_alpha(returns, benchmark, window: 'int' = 63, periods_per_year: 'int' = 252)` | Rolling annualized alpha against the benchmark. |
| `rolling_beta` | `rolling_beta(returns, benchmark, window: 'int' = 63)` | Rolling OLS beta against the benchmark. |
| `rolling_calmar` | `rolling_calmar(returns, window: 'int' = 126, periods_per_year: 'int' = 252)` | Rolling Calmar: window mean return over the window max drawdown. |
| `rolling_information_ratio` | `rolling_information_ratio(returns, benchmark, window: 'int' = 63, periods_per_year: 'int' = 252)` | Rolling annualized information ratio. |
| `rolling_max_drawdown` | `rolling_max_drawdown(returns, window: 'int' = 126)` | Rolling worst drawdown within the trailing window. |
| `rolling_sharpe` | `rolling_sharpe(returns, window: 'int' = 63, periods_per_year: 'int' = 252)` | Rolling annualized Sharpe ratio. |
| `rolling_sortino` | `rolling_sortino(returns, window: 'int' = 63, periods_per_year: 'int' = 252)` | Rolling annualized Sortino ratio. |
| `rolling_volatility` | `rolling_volatility(returns, window: 'int' = 63, periods_per_year: 'int' = 252)` | Rolling annualized volatility. |
| `sharpe_ratio` | `sharpe_ratio(returns, risk_free: 'float' = 0.0, periods_per_year: 'int' = 252)` | Annualized Sharpe ratio (per-period risk-free rate subtracted). |
| `sortino_ratio` | `sortino_ratio(returns, mar: 'float' = 0.0, periods_per_year: 'int' = 252)` | Annualized Sortino ratio. |
| `sterling_ratio` | `sterling_ratio(returns, periods_per_year: 'int' = 252)` | CAGR divided by the average drawdown magnitude. |
| `strategy_metrics` | `strategy_metrics(returns, benchmark=None, periods_per_year: 'int' = 252, risk_free: 'float' = 0.0)` | Compact metric bundle for a strategy return stream. |
| `tail_ratio` | `tail_ratio(returns, upper_q: 'float' = 0.95, lower_q: 'float' = 0.05)` | Ratio of the upper tail to the absolute lower tail at given quantiles. |
| `time_in_market` | `time_in_market(positions)` | Fraction of bars in the market (alias of exposure). |
| `total_return` | `total_return(returns)` | Compounded total return over the sample. |
| `tracking_error` | `tracking_error(returns, benchmark, periods_per_year: 'int' = 252)` | Annualized standard deviation of excess returns. |
| `trade_count` | `trade_count(trades)` | Number of trades (finite entries). |
| `trades_from_positions` | `trades_from_positions(positions, returns)` | Extract trades from a position series. |
| `treynor_ratio` | `treynor_ratio(returns, benchmark, risk_free: 'float' = 0.0, periods_per_year: 'int' = 252)` | Annualized excess return per unit of beta. |
| `turnover` | `turnover(positions)` | Total absolute position change over the sample. |
| `ulcer_index` | `ulcer_index(returns)` | Ulcer index: RMS of the drawdown series (as positive percent). |
| `underwater_series` | `underwater_series(returns)` | Alias of the drawdown series (equity below its high-water mark). |
| `up_capture` | `up_capture(returns, benchmark)` | Average up-bar return divided by the benchmark's over up bars. |
| `var_historical` | `var_historical(returns, level: 'float' = 0.05)` | Historical VaR at ``level`` (negative number for a loss). |
| `var_parametric` | `var_parametric(returns, level: 'float' = 0.05)` | Gaussian parametric VaR at ``level``. |
| `win_rate` | `win_rate(trades)` | Fraction of trades with positive PnL. |
| `yearly_returns` | `yearly_returns(returns)` | Compounded calendar-year returns. |

### `quant.risk_ext`

Risk analytics: tail/VaR, portfolio risk contributions, allocation, stress tests, sizing, regime and correlation diagnostics.

| Name | Signature | Description |
| --- | --- | --- |
| `absorption_ratio` | `absorption_ratio(cov, n_factors: 'int' = 1)` | Share of total variance explained by the top ``n_factors`` eigenvalues. |
| `atr_position_size` | `atr_position_size(account_size: 'float', risk_pct: 'float', atr_value: 'float', atr_multiple: 'float' = 2.0)` | Units sized so ``atr_multiple`` ATRs of adverse move risk ``risk_pct``. |
| `atr_stop` | `atr_stop(close: 'pd.Series', high: 'pd.Series', low: 'pd.Series', period: 'int' = 14, multiple: 'float' = 2.0, direction: 'int' = 1)` | ATR stop level below (long) or above (short) the close. |
| `average_pairwise_correlation` | `average_pairwise_correlation(returns)` | Mean off-diagonal pairwise correlation of an asset return matrix. |
| `budget_scale_weights` | `budget_scale_weights(weights, budgets, cov)` | Rescale weights so realized risk contributions match target budgets. |
| `chandelier_stop` | `chandelier_stop(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', period: 'int' = 22, multiple: 'float' = 3.0)` | Chandelier exit for long positions: rolling high minus ``multiple`` ATR. |
| `component_risk_contribution` | `component_risk_contribution(weights, cov)` | Component risk contribution: weight times marginal risk contribution. |
| `correlation_breakdown_flag` | `correlation_breakdown_flag(returns, market_returns, short_window: 'int' = 21, long_window: 'int' = 126, threshold: 'float' = 0.5)` | Flag bars where short-run correlation falls well below long-run correlation. |
| `correlation_dispersion` | `correlation_dispersion(returns)` | Standard deviation of pairwise correlations (concentration of co-movement). |
| `covariance_eigenvalues` | `covariance_eigenvalues(cov)` | Eigenvalues of a covariance matrix, sorted descending. |
| `diversification_ratio` | `diversification_ratio(weights, cov, vols=None)` | Weighted average volatility divided by portfolio volatility. |
| `dollar_hedge_ratio` | `dollar_hedge_ratio(asset_exposure: 'float', asset_price: 'float', hedge_price: 'float', beta_value: 'float')` | Units of the hedge instrument per unit of asset exposure. |
| `downside_beta` | `downside_beta(returns, benchmark)` | Beta computed only over benchmark-down bars. |
| `downside_correlation` | `downside_correlation(returns, market_returns)` | Correlation measured only over market-down bars. |
| `drawdown_recovery_bars` | `drawdown_recovery_bars(returns)` | Mean number of bars from a drawdown trough back to a new equity high. |
| `drawdown_speed` | `drawdown_speed(returns)` | Max drawdown magnitude divided by the drawdown duration in bars. |
| `effective_number_of_bets` | `effective_number_of_bets(weights, cov=None)` | Effective number of independent risk bets (1 / HHI of risk contributions). |
| `effective_rank` | `effective_rank(cov)` | Entropy-based effective rank of the covariance matrix. |
| `equal_risk_contribution` | `equal_risk_contribution(returns, iters: 'int' = 200, tol: 'float' = 1e-08)` | Iterative equal-risk-contribution (risk parity) weights. |
| `ewma_volatility` | `ewma_volatility(returns, halflife: 'int' = 21, periods_per_year: 'int' = 252)` | Annualized EWMA volatility. |
| `expected_shortfall_portfolio` | `expected_shortfall_portfolio(weights, asset_returns, level: 'float' = 0.05)` | Historical expected shortfall of the portfolio return stream. |
| `expected_tail_loss` | `expected_tail_loss(returns, tail_pct: 'float' = 0.05)` | Mean loss beyond the worst ``tail_pct`` of observations (positive number). |
| `fractional_kelly` | `fractional_kelly(kelly_value: 'float', fraction: 'float' = 0.5)` | Fractional Kelly (``fraction`` of the full Kelly bet). |
| `gross_exposure` | `gross_exposure(weights)` | Sum of absolute weights. |
| `hedge_effectiveness` | `hedge_effectiveness(asset, hedge, hedge_ratio_value: 'float | None' = None)` | R-squared of the hedged portfolio variance versus the unhedged asset. |
| `hedge_ratio` | `hedge_ratio(asset, hedge)` | OLS hedge ratio: beta of the asset on the hedging instrument. |
| `hhi_concentration` | `hhi_concentration(weights)` | Herfindahl-Hirschman concentration index of (absolute) weights. |
| `historical_stress_windows` | `historical_stress_windows(weights, asset_returns, window: 'int' = 21, top_n: 'int' = 5)` | Worst overlapping historical windows of the portfolio return series. |
| `inverse_vol_weights` | `inverse_vol_weights(returns)` | Weights proportional to inverse volatility. |
| `kelly_from_moments` | `kelly_from_moments(mean_return: 'float', std_return: 'float', risk_free: 'float' = 0.0)` | Continuous Kelly fraction from mean/std moments. |
| `leverage_ratio` | `leverage_ratio(weights)` | Gross exposure divided by net capital (alias of gross exposure). |
| `marginal_risk_contribution` | `marginal_risk_contribution(weights, cov)` | Marginal contribution of each asset to portfolio volatility. |
| `max_diversification_weights` | `max_diversification_weights(cov)` | Closed-form maximum-diversification weights w ~ inv(cov) @ sigma. |
| `max_loss_streak` | `max_loss_streak(returns)` | Longest run of consecutive negative bars. |
| `min_variance_weights` | `min_variance_weights(cov)` | Closed-form global minimum-variance weights. |
| `minimum_variance_hedge` | `minimum_variance_hedge(asset, hedge)` | Minimum-variance hedge ratio: cov(asset, hedge) / var(hedge). |
| `net_exposure` | `net_exposure(weights)` | Sum of signed weights. |
| `portfolio_beta` | `portfolio_beta(weights, betas)` | Weighted average of asset betas. |
| `portfolio_volatility` | `portfolio_volatility(weights, cov)` | Portfolio volatility from weights and a covariance matrix. |
| `position_limit_scale` | `position_limit_scale(weights, max_gross: 'float' = 1.0)` | Scale factor (<= 1) that caps gross exposure at ``max_gross``. |
| `position_size_by_risk` | `position_size_by_risk(account_size: 'float', risk_pct: 'float', stop_distance_price: 'float')` | Units to trade so a stop at ``stop_distance_price`` risks ``risk_pct``. |
| `return_quantile_profile` | `return_quantile_profile(returns, quantiles: 'tuple[float, ...]' = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))` | Quantile profile of the return distribution. |
| `risk_budget_weights` | `risk_budget_weights(cov, budgets=None, iters: 'int' = 200, tol: 'float' = 1e-08)` | Multiplicative-update weights that match per-asset risk budgets. |
| `risk_contribution_pct` | `risk_contribution_pct(weights, cov)` | Percentage risk contribution per asset (sums to 1). |
| `rolling_correlation_to_market` | `rolling_correlation_to_market(returns, market_returns, window: 'int' = 63)` | Rolling correlation of the strategy with the market. |
| `rolling_cvar` | `rolling_cvar(returns, window: 'int' = 250, level: 'float' = 0.05)` | Rolling historical CVaR (expected shortfall). |
| `rolling_hedge_ratio` | `rolling_hedge_ratio(asset, hedge, window: 'int' = 63)` | Rolling minimum-variance hedge ratio. |
| `rolling_var` | `rolling_var(returns, window: 'int' = 250, level: 'float' = 0.05)` | Rolling historical VaR. |
| `scenario_shock_analysis` | `scenario_shock_analysis(weights, shocks)` | Per-asset contribution to a shock scenario plus the total impact. |
| `stop_distance_to_volatility` | `stop_distance_to_volatility(stop_distance_pct: 'float', volatility: 'float', periods_per_year: 'int' = 252)` | Stop distance expressed as a multiple of annualized volatility. |
| `stress_scenario_impact` | `stress_scenario_impact(weights, asset_returns, scenario=None)` | Portfolio return under a per-asset shock vector. |
| `tail_beta` | `tail_beta(returns, benchmark, tail_pct: 'float' = 0.1)` | Beta computed over the worst ``tail_pct`` of benchmark bars. |
| `tail_index_hill` | `tail_index_hill(returns, k: 'int' = 50)` | Hill estimator of the tail index from the k largest losses. |
| `time_underwater` | `time_underwater(returns)` | Fraction of bars spent below a prior equity peak. |
| `trailing_stop` | `trailing_stop(close: 'pd.Series', pct: 'float' = 0.05, direction: 'int' = 1)` | Percentage trailing stop from the running extreme. |
| `turbulence_index` | `turbulence_index(returns)` | Mahalanobis distance of each bar's cross-asset returns from the mean. |
| `upside_beta` | `upside_beta(returns, benchmark)` | Beta computed only over benchmark-up bars. |
| `var_bootstrap_ci` | `var_bootstrap_ci(returns, level: 'float' = 0.05, n_boot: 'int' = 500, alpha: 'float' = 0.05, seed: 'int' = 42)` | Bootstrap confidence interval for the historical VaR estimate. |
| `var_scale` | `var_scale(returns, horizon: 'int' = 10, level: 'float' = 0.05)` | Historical VaR scaled to ``horizon`` bars by the square-root-of-time rule. |
| `vol_spike_flag` | `vol_spike_flag(returns, window: 'int' = 21, z_thresh: 'float' = 2.0)` | Boolean flags where rolling volatility deviates above its history. |
| `volatility_beta` | `volatility_beta(returns, benchmark, window: 'int' = 63)` | Ratio of rolling volatility to the benchmark's over ``window`` bars. |
| `volatility_cone` | `volatility_cone(returns, windows: 'tuple[int, ...]' = (5, 10, 21, 63, 126, 252), quantiles: 'tuple[float, ...]' = (0.05, 0.25, 0.5, 0.75, 0.95), periods_per_year: 'int' = 252)` | Volatility cone: quantiles of annualized rolling volatility per window. |
| `volatility_regime` | `volatility_regime(returns, window: 'int' = 21, low_q: 'float' = 0.33, high_q: 'float' = 0.66)` | Regime labels from rolling volatility vs its expanding quantiles: -1/0/+1. |
| `volatility_regime_z` | `volatility_regime_z(returns, window: 'int' = 21, lookback: 'int' = 252)` | Z-score of current rolling volatility within its trailing history. |
| `volatility_target_position` | `volatility_target_position(target_vol: 'float', realized_vol: 'float', max_leverage: 'float' = 1.0)` | Position scale that targets ``target_vol`` given realized volatility. |
| `worst_window_return` | `worst_window_return(returns, window: 'int' = 21)` | Worst compounded return over any rolling window of ``window`` bars. |

### `quant.portfolio_ext`

Portfolio construction: weight estimators, covariance estimators, rebalancing, attribution and reporting.

| Name | Signature | Description |
| --- | --- | --- |
| `active_returns` | `active_returns(weights, asset_returns, benchmark_weights)` | Active (portfolio minus benchmark) return stream. |
| `active_weights` | `active_weights(weights, benchmark_weights)` | Difference between portfolio and benchmark weights. |
| `align_returns` | `align_returns(*frames)` | Align several return frames on their common index (inner join). |
| `attribution_table` | `attribution_table(weights, asset_returns)` | Weights, mean return, total contribution, and risk contribution per asset. |
| `black_litterman_weights` | `black_litterman_weights(returns, market_weights, views=None, view_confidence: 'float' = 0.5, tau: 'float' = 0.05, risk_aversion: 'float' = 2.0)` | Black-Litterman posterior weights from market-implied equilibrium returns. |
| `calendar_rebalance_returns` | `calendar_rebalance_returns(weights, asset_returns, freq: 'str' = 'ME', cost: 'float' = 0.0)` | Portfolio returns with calendar rebalancing at each period end. |
| `concentration_metrics` | `concentration_metrics(weights)` | Concentration diagnostics: max weight, top-5 share, HHI, effective number. |
| `contribution_by_asset` | `contribution_by_asset(weights, asset_returns)` | Total compounded-times-weight contribution per asset (sum of contributions). |
| `contribution_series` | `contribution_series(weights, asset_returns)` | Per-period contribution of each asset to the portfolio return. |
| `correlation_to_covariance` | `correlation_to_covariance(corr, vols)` | Convert a correlation matrix plus volatilities to a covariance matrix. |
| `covariance_to_correlation` | `covariance_to_correlation(cov)` | Convert a covariance matrix to a correlation matrix. |
| `diversification_benefit` | `diversification_benefit(weights, asset_returns, periods_per_year: 'int' = 252)` | Weighted-average asset volatility minus portfolio volatility. |
| `downside_risk_portfolio` | `downside_risk_portfolio(weights, asset_returns, mar: 'float' = 0.0, periods_per_year: 'int' = 252)` | Annualized downside deviation of the portfolio return stream. |
| `drift_weights` | `drift_weights(weights, asset_returns)` | Weights after buy-and-hold drift over the sample (no rebalancing). |
| `effective_number_of_assets` | `effective_number_of_assets(weights)` | Effective number of holdings (1 / Herfindahl index of weights). |
| `efficient_frontier` | `efficient_frontier(returns, n_points: 'int' = 20)` | Efficient frontier points (annualized risk/return) via two-fund separation. |
| `equal_weight` | `equal_weight(assets)` | Equal weights across assets (accepts a list, index, or DataFrame). |
| `exponentially_weighted_covariance` | `exponentially_weighted_covariance(returns, halflife: 'int' = 63)` | Exponentially weighted covariance matrix. |
| `global_min_variance_weights` | `global_min_variance_weights(returns)` | Global minimum-variance weights from the sample covariance. |
| `hierarchical_risk_parity` | `hierarchical_risk_parity(returns, n_clusters: 'int' = 2, iters: 'int' = 200)` | Simplified hierarchical risk parity. |
| `inverse_variance_weights` | `inverse_variance_weights(returns)` | Weights proportional to inverse variance. |
| `inverse_volatility_weights` | `inverse_volatility_weights(returns)` | Weights proportional to inverse volatility. |
| `ledoit_wolf_covariance` | `ledoit_wolf_covariance(returns)` | Ledoit-Wolf shrunk covariance matrix. |
| `long_only_weights` | `long_only_weights(weights)` | Clip negatives to zero and renormalize to sum to one. |
| `marginal_contribution_to_var` | `marginal_contribution_to_var(weights, asset_returns, level: 'float' = 0.05)` | Parametric component VaR per asset (sums to total parametric VaR). |
| `market_cap_weights` | `market_cap_weights(caps)` | Capitalization weights. |
| `max_diversification_portfolio` | `max_diversification_portfolio(returns)` | Maximum-diversification weights. |
| `max_sharpe_weights` | `max_sharpe_weights(returns, risk_free: 'float' = 0.0)` | Tangency-portfolio weights (closed form, normalized to unit gross). |
| `mean_variance_weights` | `mean_variance_weights(returns, risk_aversion: 'float' = 1.0)` | Unconstrained mean-variance weights (risk aversion scaled, gross-normalized). |
| `normalize_weights` | `normalize_weights(weights)` | Scale weights to sum to one (absolute sum when the net is zero). |
| `performance_summary` | `performance_summary(weights, asset_returns, benchmark=None, periods_per_year: 'int' = 252)` | Metric bundle of a static-weight portfolio (see ``strategy_metrics``). |
| `portfolio_drawdown` | `portfolio_drawdown(weights, asset_returns)` | Drawdown series of a static-weight portfolio. |
| `portfolio_equity` | `portfolio_equity(weights, asset_returns, initial: 'float' = 1.0)` | Compounded equity curve of a static-weight portfolio. |
| `portfolio_returns` | `portfolio_returns(weights, asset_returns)` | Weighted portfolio return series from static weights. |
| `rebalance_dates` | `rebalance_dates(index, freq: 'str' = 'ME')` | Trading dates on which a periodic rebalance occurs. |
| `rebalance_schedule` | `rebalance_schedule(weights, asset_returns, freq: 'str' = 'ME', band: 'float | None' = None)` | Target weights on each rebalance date given calendar and/or drift bands. |
| `rebalancing_turnover` | `rebalancing_turnover(schedule: 'pd.DataFrame')` | One-way turnover between consecutive target-weight rows of a schedule. |
| `returns_matrix` | `returns_matrix(prices)` | Per-period returns of a price matrix. |
| `risk_budget_table` | `risk_budget_table(weights, asset_returns)` | Absolute and percentage risk contributions per asset. |
| `risk_parity_weights` | `risk_parity_weights(returns, iters: 'int' = 200, tol: 'float' = 1e-08)` | Equal-risk-contribution (risk parity) weights. |
| `rolling_portfolio_volatility` | `rolling_portfolio_volatility(weights, asset_returns, window: 'int' = 63, periods_per_year: 'int' = 252)` | Rolling annualized volatility of the static-weight portfolio. |
| `sample_covariance` | `sample_covariance(returns)` | Sample covariance matrix. |
| `shrinkage_covariance` | `shrinkage_covariance(returns, shrink: 'float' = 0.2)` | Shrink the sample covariance toward its diagonal. |
| `shrinkage_weights` | `shrinkage_weights(returns, shrink: 'float' = 0.5, risk_aversion: 'float' = 1.0)` | Blend mean-variance weights toward equal weights by ``shrink``. |
| `target_volatility_weights` | `target_volatility_weights(weights, returns, target_vol: 'float' = 0.1, periods_per_year: 'int' = 252)` | Scale weights (leverage) so realized portfolio volatility hits the target. |
| `threshold_rebalance_returns` | `threshold_rebalance_returns(weights, asset_returns, band: 'float' = 0.05, cost: 'float' = 0.0)` | Portfolio returns with drift-band (threshold) rebalancing. |
| `turnover_penalized_weights` | `turnover_penalized_weights(weights, previous_weights, penalty: 'float' = 0.5)` | Partial-adjust weights toward the previous allocation by ``penalty``. |
| `volatility_scaled_weights` | `volatility_scaled_weights(returns, target_vol: 'float' = 0.1, periods_per_year: 'int' = 252)` | Inverse-vol weights rescaled so the portfolio targets ``target_vol``. |
| `weight_cap` | `weight_cap(weights, max_weight: 'float' = 0.25, iters: 'int' = 50, tol: 'float' = 1e-08)` | Cap individual weights at ``max_weight``, redistributing excess to the rest. |
| `weights_from_scores` | `weights_from_scores(scores, long_short: 'bool' = True)` | Convert cross-sectional scores to weights (demeaned and gross-normalized). |

### `quant.strategies_ext`

66 trading strategies across trend, mean-reversion, momentum, volatility, volume, pattern and composite families.

| Name | Signature | Description |
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

### `quant.options`

Options analytics: Black-Scholes pricing and Greeks, implied volatility, binomial American pricing, vol surface helpers, strategy payoffs, risk-neutral density.

| Name | Signature | Description |
| --- | --- | --- |
| `bear_call_spread` | `bear_call_spread(low_strike, high_strike, qty=1)` | Bear call spread: short low-strike call, long high-strike call. |
| `bear_put_spread` | `bear_put_spread(high_strike, low_strike, qty=1)` | Bear put spread: long high-strike put, short low-strike put. |
| `binomial_tree` | `binomial_tree(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', steps: 'int' = 200, option_type: 'str' = 'call', american: 'bool' = False, q: 'float' = 0.0)` | Cox-Ross-Rubinstein binomial price (supports early American exercise). |
| `black_scholes_greeks` | `black_scholes_greeks(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0, per_vol_point: 'bool' = True)` | Bundle of primary Greeks; vega/theta optionally scaled per vol-point/day. |
| `black_scholes_price` | `black_scholes_price(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | European Black-Scholes price with continuous dividend yield ``q``. |
| `box_spread` | `box_spread(low_strike, high_strike, qty=1)` | Box spread using a call and a put spread between two strikes. |
| `breeden_litzenberger` | `breeden_litzenberger(call_prices, strikes, t: 'float', r: 'float')` | Risk-neutral density from call prices: ``e^{rT} d^2C/dK^2``. |
| `bs_delta` | `bs_delta(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | Option delta (spot sensitivity). |
| `bs_gamma` | `bs_gamma(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', q: 'float' = 0.0)` | Option gamma (delta sensitivity); identical for calls and puts. |
| `bs_rho` | `bs_rho(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | Option rho (rate sensitivity) per 1.0 change in ``r``. |
| `bs_theta` | `bs_theta(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | Option theta per year (divide by 365 for per-day). |
| `bs_vega` | `bs_vega(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', q: 'float' = 0.0)` | Option vega for a 1.0 change in volatility (divide by 100 for per-vol-point). |
| `bull_call_spread` | `bull_call_spread(low_strike, high_strike, qty=1)` | Bull call spread: long low-strike call, short high-strike call. |
| `bull_put_spread` | `bull_put_spread(high_strike, low_strike, qty=1)` | Bull put spread: short high-strike put, long low-strike put. |
| `butterfly` | `butterfly(low_strike, mid_strike, high_strike, qty=1)` | Long call butterfly: +1/-2/+1 at increasing strikes. |
| `collar` | `collar(put_strike, call_strike, qty=1)` | Collar: long protective put + short covered call. |
| `condor` | `condor(put_lo, put_hi, call_lo, call_hi, qty=1)` | Long condor: four calls spaced across strikes. |
| `covered_call` | `covered_call(strike, qty=1)` | Covered call: short call against long stock (stock leg implied). |
| `delta_hedge_shares` | `delta_hedge_shares(delta: 'float', contracts: 'int' = 1, contract_size: 'int' = 100)` | Shares required to delta-hedge ``contracts`` (negative delta offsets longs). |
| `delta_to_strike` | `delta_to_strike(spot: 'float', t: 'float', r: 'float', sigma: 'float', delta: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | Strike that produces a given BS delta. |
| `discount_factor` | `discount_factor(r: 'float', t: 'float')` | Continuous-compounding discount factor ``exp(-r t)``. |
| `expected_move` | `expected_move(spot: 'float', sigma: 'float', t: 'float', z: 'float' = 1.0)` | One-standard-deviation (or ``z``-sigma) price move over horizon ``t``. |
| `forward_price` | `forward_price(spot: 'float', r: 'float', t: 'float', q: 'float' = 0.0)` | No-arbitrage forward price ``S e^{(r-q)t}``. |
| `gamma_scalp_pnl` | `gamma_scalp_pnl(gamma: 'float', spot_move: 'float', theta_daily: 'float', days: 'float' = 1.0, contract_size: 'int' = 100)` | Approximate gamma-scalping PnL: ``0.5 gamma (dS)^2 - theta`` per contract block. |
| `greeks_series` | `greeks_series(spot: 'pd.Series', strike, t, r: 'float', sigma, option_type: 'str' = 'call', q: 'float' = 0.0)` | Vectorized Greeks across a spot series (strike/sigma broadcastable). |
| `implied_vol_bisection` | `implied_vol_bisection(market_price: 'float', spot: 'float', strike: 'float', t: 'float', r: 'float', option_type: 'str' = 'call', q: 'float' = 0.0, tol: 'float' = 1e-08, max_iter: 'int' = 200)` | Implied volatility by bisection on a wide volatility bracket. |
| `implied_volatility` | `implied_volatility(market_price: 'float', spot: 'float', strike: 'float', t: 'float', r: 'float', option_type: 'str' = 'call', q: 'float' = 0.0, tol: 'float' = 1e-08, max_iter: 'int' = 100)` | Implied volatility by Newton-Raphson with a bisection fallback. |
| `intrinsic_value` | `intrinsic_value(spot: 'float', strike: 'float', option_type: 'str' = 'call')` | Exercise value of the option. |
| `inv_norm_cdf` | `inv_norm_cdf(p)` | Inverse standard normal CDF (Acklam rational approximation). |
| `iron_butterfly` | `iron_butterfly(put_strike, call_strike, put_wing, call_wing, qty=1)` | Short iron butterfly: short straddle with long OTM wings. |
| `iron_condor` | `iron_condor(put_lo, put_hi, call_lo, call_hi, qty=1)` | Short iron condor: short inner strangle, long outer wings. |
| `log_moneyness` | `log_moneyness(spot: 'float', strike: 'float', sigma: 'float', t: 'float')` | Standardized log-moneyness ``ln(S/K) / (sigma sqrt(t))``. |
| `moneyness` | `moneyness(spot: 'float', strike: 'float')` | Simple moneyness ``S / K``. |
| `norm_cdf` | `norm_cdf(x)` | Standard normal CDF (vectorized). |
| `norm_pdf` | `norm_pdf(x)` | Standard normal PDF (vectorized). |
| `option_elasticity` | `option_elasticity(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | Percent option move per percent underlying move (delta * S / price). |
| `option_price` | `option_price(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0, engine: 'str' = 'bs', **kwargs)` | Dispatch to ``bs`` or ``binomial`` pricing. |
| `parity_gap` | `parity_gap(call_price: 'float', put_price: 'float', spot: 'float', strike: 'float', t: 'float', r: 'float', q: 'float' = 0.0)` | Observed put-call parity violation ``C - P - (S e^{-q t} - K e^{-r t})``. |
| `price_series` | `price_series(spot: 'pd.Series', strike, t, r: 'float', sigma, option_type: 'str' = 'call', q: 'float' = 0.0)` | Vectorized Black-Scholes prices across a spot series. |
| `prob_itm` | `prob_itm(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | Risk-neutral probability of finishing in the money. |
| `prob_otm` | `prob_otm(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | Risk-neutral probability of finishing out of the money. |
| `prob_touch` | `prob_touch(spot: 'float', barrier: 'float', t: 'float', sigma: 'float', drift: 'float' = 0.0)` | Approximate probability a driftless lognormal touches ``barrier`` by ``t``. |
| `profit_table` | `profit_table(legs, spot_range)` | Payoff, premium, and net PnL per spot level. |
| `protective_put` | `protective_put(strike, qty=1)` | Protective put: long put against long stock. |
| `put_call_parity` | `put_call_parity(call_price: 'float', spot: 'float', strike: 'float', t: 'float', r: 'float', q: 'float' = 0.0)` | Theoretical put price implied by call price via put-call parity. |
| `ratio_spread` | `ratio_spread(low_strike, high_strike, ratio=2, option_type='call', qty=1)` | Ratio spread: long 1x low strike, short ``ratio`` high strike. |
| `risk_neutral_density` | `risk_neutral_density(prices, strikes, t: 'float', r: 'float')` | Alias of :func:`breeden_litzenberger`. |
| `risk_reversal` | `risk_reversal(put_strike, call_strike, qty=1)` | Risk reversal: long OTM call + short OTM put. |
| `straddle` | `straddle(strike, premium_call=0.0, premium_put=0.0, qty=1)` | Long straddle: long call + long put at the same strike. |
| `strangle` | `strangle(put_strike, call_strike, premium_call=0.0, premium_put=0.0, qty=1)` | Long strangle: long OTM put + long OTM call. |
| `strategy_breakevens` | `strategy_breakevens(legs, spot_range)` | Approximate breakeven spots by locating payoff sign changes. |
| `strategy_max_loss` | `strategy_max_loss(legs, spot_range)` | Minimum payoff minus net premium over the spot range. |
| `strategy_max_profit` | `strategy_max_profit(legs, spot_range)` | Maximum payoff minus net premium over the spot range. |
| `strategy_payoff` | `strategy_payoff(legs, spot_range)` | Expiry payoff of a leg list across a spot range (premium excluded). |
| `strategy_value` | `strategy_value(legs, spot_range, t, r, sigma, q: 'float' = 0.0)` | Strategy value across spot range using Black-Scholes (premiums excluded). |
| `strike_to_delta` | `strike_to_delta(spot: 'float', strike: 'float', t: 'float', r: 'float', sigma: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | BS delta for a given strike. |
| `synthetic_forward` | `synthetic_forward(strike, qty=1)` | Synthetic long forward: long call + short put at the same strike. |
| `time_value` | `time_value(price: 'float', spot: 'float', strike: 'float', option_type: 'str' = 'call')` | Option price minus intrinsic value. |
| `variance_swap_strike` | `variance_swap_strike(implied_vols, weights=None)` | Approximate variance-swap fair strike ``sqrt(E[sigma^2])`` from a vol strip. |
| `vix_proxy` | `vix_proxy(implied_vols: 'pd.Series')` | Simple 30-day VIX-style proxy: mean of near-the-money implied vols. |
| `vol_skew` | `vol_skew(vols: 'pd.Series', strikes: 'pd.Series | None' = None)` | 25-delta-style skew proxy: low-strike IV minus high-strike IV. |
| `vol_smile` | `vol_smile(prices, spot: 'float', strikes, t: 'float', r: 'float', option_type: 'str' = 'call', q: 'float' = 0.0)` | Implied-volatility smile for a range of strikes. |
| `vol_term_structure` | `vol_term_structure(vols_by_expiry: 'dict')` | Sort implied vols by expiry (keys = tenor in years or ordered labels). |

### `quant.execution_ext`

Execution and TCA: TWAP/VWAP/POV scheduling, arrival/interval benchmarks, implementation shortfall, spread/microprice estimators, order-flow imbalance, impact approximations.

| Name | Signature | Description |
| --- | --- | --- |
| `abdi_ranaldo_spread` | `abdi_ranaldo_spread(open_, high, low, close)` | Abdi-Ranaldo close-high-low bid-ask spread estimator (decimal). |
| `adverse_selection_cost` | `adverse_selection_cost(trade_price: 'float', mid_at_exec: 'float', mid_later: 'float', side: 'str' = 'buy')` | Effective minus realized spread (the adverse-selection component). |
| `amihud_impact` | `amihud_impact(returns, dollar_volume)` | Amihud-style illiquidity: absolute return per unit dollar volume. |
| `arrival_price` | `arrival_price(prices, index: 'int' = 0)` | Price at order arrival (decision) time. |
| `back_loaded_schedule` | `back_loaded_schedule(order_size: 'float', n_slices: 'int', ramp: 'float' = 2.0)` | Exponentially back-loaded slicing (trade faster late). |
| `bar_vwap` | `bar_vwap(bars: 'pd.DataFrame', price_col: 'str' = 'close', volume_col: 'str' = 'volume')` | Volume-weighted average price over bars. |
| `bps_to_return` | `bps_to_return(bps: 'float')` | Convert basis points to a decimal return. |
| `compare_execution` | `compare_execution(results: 'dict')` | Rank an {algo: trades-DataFrame} mapping by mean slippage and total cost. |
| `corwin_schultz_spread` | `corwin_schultz_spread(high, low)` | Corwin-Schultz high-low bid-ask spread estimator (decimal). |
| `cost_curve` | `cost_curve(order_sizes, adv: 'float', sigma: 'float', eta: 'float' = 0.1)` | Impact cost (bps) and participation across a range of order sizes. |
| `depth_weighted_price` | `depth_weighted_price(bids, asks)` | Depth-weighted mid using ``[(price, size), ...]`` ladders for bids and asks. |
| `effective_spread` | `effective_spread(trade_price: 'float', mid: 'float', side: 'str' = 'buy')` | Effective spread: ``2 * |trade - mid|`` signed cost. |
| `execution_cost_bps` | `execution_cost_bps(arrival_pr: 'float', exec_pr: 'float', side: 'str' = 'buy')` | Signed execution cost in bps versus arrival price (positive = slippage). |
| `execution_report` | `execution_report(trades: 'pd.DataFrame')` | Markdown TCA report from a trades table. |
| `expected_cost` | `expected_cost(participation: 'float', spread_bps: 'float', sigma: 'float', horizon: 'float', impact_coeff: 'float' = 0.1)` | Heuristic expected cost in bps: half-spread + temporary impact. |
| `fill_probability` | `fill_probability(distance: 'float', spread: 'float', volatility: 'float', horizon: 'float')` | Heuristic probability a passive order at ``distance`` fills within ``horizon``. |
| `fill_rate` | `fill_rate(order_size: 'float', filled: 'float')` | Fraction of the order that was filled. |
| `front_loaded_schedule` | `front_loaded_schedule(order_size: 'float', n_slices: 'int', decay: 'float' = 2.0)` | Exponentially front-loaded slicing (trade faster early). |
| `implementation_shortfall` | `implementation_shortfall(decision_price: 'float', avg_exec_price: 'float', side: 'str', size: 'float', unfilled: 'float' = 0.0, close_price: 'float | None' = None)` | Implementation shortfall in dollars: executed slippage plus opportunity cost. |
| `interval_vwap` | `interval_vwap(bars: 'pd.DataFrame', start: 'int' = 0, end: 'int | None' = None, price_col: 'str' = 'close', volume_col: 'str' = 'volume')` | VWAP over a bar interval ``[start, end)``. |
| `kyle_lambda` | `kyle_lambda(prices, volume, window: 'int' = 50)` | Rolling Kyle's lambda: price impact per unit signed order flow. |
| `lee_ready_classification` | `lee_ready_classification(trade_price, bid, ask)` | Lee-Ready trade classification using the quote midpoint rule. |
| `limit_fill_probability` | `limit_fill_probability(market_orders_per_bar: 'float', ahead_of_queue: 'float', bars: 'float')` | Approximate fill probability for a limit order given queue depth ahead. |
| `linear_schedule` | `linear_schedule(order_size: 'float', n_slices: 'int')` | Equal-sized slicing of ``order_size`` across ``n_slices``. |
| `market_impact_estimate` | `market_impact_estimate(order_notional: 'float', adv_notional: 'float', sigma: 'float', eta: 'float' = 0.1)` | Square-root market-impact estimate in bps given participation and daily vol. |
| `microprice` | `microprice(bid: 'float', ask: 'float', bid_size: 'float', ask_size: 'float')` | Size-weighted microprice leaning toward the heavier book side. |
| `mid_price` | `mid_price(bid: 'float', ask: 'float')` | Midpoint of the best bid and offer. |
| `notional_from_shares` | `notional_from_shares(shares: 'float', price: 'float')` | Dollar notional of a share quantity at ``price``. |
| `opportunity_cost` | `opportunity_cost(decision_price: 'float', close_price: 'float', side: 'str', unfilled: 'float')` | Cost of shares never filled, marked versus the decision price. |
| `optimal_execution_horizon` | `optimal_execution_horizon(order_size: 'float', adv: 'float', participation: 'float' = 0.1)` | Days to complete ``order_size`` at ``participation`` of average daily volume. |
| `order_book_imbalance` | `order_book_imbalance(bid_size: 'float', ask_size: 'float')` | Book imbalance in [-1, 1]; positive = bid-heavy. |
| `order_size_from_notional` | `order_size_from_notional(notional: 'float', price: 'float')` | Shares required to reach a dollar notional at ``price``. |
| `participation_curve` | `participation_curve(volume_profile, participation: 'float' = 0.1)` | Tradeable size per bar at a fixed participation rate. |
| `participation_rate` | `participation_rate(filled: 'float', market_volume: 'float')` | Fraction of market volume traded. |
| `pov_schedule` | `pov_schedule(order_size: 'float', volume_forecast, participation: 'float' = 0.1)` | Percent-of-volume schedule capped by ``participation`` of forecast volume. |
| `price_improvement` | `price_improvement(exec_price: 'float', quote_price: 'float', side: 'str' = 'buy')` | Positive when executed better than the prevailing quote. |
| `queue_depth` | `queue_depth(bid_size: 'float', ask_size: 'float')` | Summary of displayed depth on both sides. |
| `quoted_spread` | `quoted_spread(bid: 'float', ask: 'float')` | Absolute quoted spread. |
| `realized_spread` | `realized_spread(trade_price: 'float', mid_at_exec: 'float', mid_later: 'float', side: 'str' = 'buy')` | Realized spread capturing adverse selection (uses a later mid). |
| `relative_spread` | `relative_spread(bid: 'float', ask: 'float')` | Quoted spread divided by mid (a decimal fraction). |
| `return_to_bps` | `return_to_bps(ret: 'float')` | Convert a decimal return to basis points. |
| `rolling_participation` | `rolling_participation(fills, market_volume, window: 'int' = 20)` | Rolling participation rate of fills versus market volume. |
| `slice_order` | `slice_order(order_size: 'float', n_slices: 'int', ramp: 'str' = 'flat', decay: 'float' = 2.0)` | Dispatch to a slicing style: ``flat``, ``front``, or ``back``. |
| `slippage_decomposition` | `slippage_decomposition(arrival_pr: 'float', exec_pr: 'float', interval_vwap: 'float', side: 'str' = 'buy')` | Split slippage into delay (arrival->interval) and trading (interval->fill) parts. |
| `tca_summary` | `tca_summary(trades: 'pd.DataFrame', arrival_col: 'str' = 'arrival_price', exec_col: 'str' = 'exec_price', size_col: 'str' = 'size', vwap_col: 'str | None' = 'vwap')` | Per-trade VWAP/arrival slippage (bps) and signed dollar cost. |
| `tick_classification` | `tick_classification(prices)` | Uptick/downtick classification: 1 buy, -1 sell, 0 unchanged. |
| `trade_imbalance` | `trade_imbalance(buy_volume: 'float', sell_volume: 'float')` | Signed trade imbalance in [-1, 1]. |
| `twap_error` | `twap_error(exec_price: 'float', benchmark_twap: 'float', side: 'str' = 'buy')` | Bps of execution versus the TWAP benchmark. |
| `twap_from_bars` | `twap_from_bars(bars: 'pd.DataFrame', price_col: 'str' = 'close')` | Simple average of the price column (TWAP over the bars). |
| `twap_schedule` | `twap_schedule(order_size: 'float', n_slices: 'int')` | Time-weighted average price schedule (flat slices over time). |
| `volume_profile` | `volume_profile(volume, n_buckets: 'int' = 10)` | Aggregate intraday/bar volume into ``n_buckets`` normalized buckets. |
| `vwap_error` | `vwap_error(exec_price: 'float', benchmark_vwap: 'float', side: 'str' = 'buy')` | Bps of execution versus the VWAP benchmark. |
| `vwap_schedule` | `vwap_schedule(order_size: 'float', volume_profile)` | Volume-weighted schedule proportional to a forecast volume profile. |
| `weighted_avg_price` | `weighted_avg_price(prices, sizes)` | Size-weighted average execution price. |

## Usage

```python
from quant import catalog, features_ext, strategies_ext, portfolio_ext

catalog.registry_counts()                 # discover everything
catalog.search('divergence')              # search names/docstrings
X = features_ext.build_feature_matrix(df) # engineered feature matrix
pos = strategies_ext.strategy_positions('sma_cross', df)
w = portfolio_ext.risk_parity_weights(returns)
```
