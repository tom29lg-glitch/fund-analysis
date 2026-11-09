"""Automatic interpretation of financial analysis results."""

import pandas as pd


def _is_significant(
    p_value: float,
    significance_level: float,
) -> bool:
    """Return whether a result is statistically significant."""
    return bool(p_value < significance_level)


def interpret_carhart_model(
    coefficient_table: pd.DataFrame,
    model_summary: pd.Series,
    significance_level: float = 0.05,
) -> list[str]:
    """Interpret the Carhart four-factor regression."""
    interpretations = []

    r_squared = model_summary["r_squared"]
    interpretations.append(
        f"The Carhart model explains "
        f"{r_squared:.1%} of the fund's monthly return variation."
    )

    alpha = coefficient_table.loc["alpha", "coefficient"]
    alpha_p_value = coefficient_table.loc[
        "alpha", "p_value"
    ]

    if _is_significant(
        alpha_p_value,
        significance_level,
    ):
        direction = "positive" if alpha > 0 else "negative"
        interpretations.append(
            f"The fund has a statistically significant "
            f"{direction} monthly alpha of {alpha:.2%} "
            f"(p-value: {alpha_p_value:.4f})."
        )
    else:
        interpretations.append(
            f"The estimated monthly alpha is {alpha:.2%}, "
            f"but it is not statistically significant "
            f"(p-value: {alpha_p_value:.4f}). "
            f"There is insufficient evidence of abnormal "
            f"risk-adjusted performance."
        )

    factor_descriptions = {
        "MKT_RF": {
            "positive": (
                "a positive exposure to the equity market"
            ),
            "negative": (
                "a negative exposure to the equity market"
            ),
        },
        "SMB": {
            "positive": (
                "a small-cap investment bias"
            ),
            "negative": (
                "a large-cap investment bias"
            ),
        },
        "HML": {
            "positive": (
                "a value investment bias"
            ),
            "negative": (
                "a growth investment bias"
            ),
        },
        "MOM": {
            "positive": (
                "a positive momentum exposure"
            ),
            "negative": (
                "a contrarian or negative momentum exposure"
            ),
        },
    }

    for factor, descriptions in factor_descriptions.items():
        coefficient = coefficient_table.loc[
            factor, "coefficient"
        ]
        p_value = coefficient_table.loc[
            factor, "p_value"
        ]

        if _is_significant(
            p_value,
            significance_level,
        ):
            direction = (
                "positive"
                if coefficient > 0
                else "negative"
            )
            interpretations.append(
                f"{factor}: the coefficient of "
                f"{coefficient:.3f} indicates "
                f"{descriptions[direction]} "
                f"(p-value: {p_value:.4f})."
            )
        else:
            interpretations.append(
                f"{factor}: the coefficient is "
                f"{coefficient:.3f}, but the exposure is "
                f"not statistically significant "
                f"(p-value: {p_value:.4f})."
            )

    return interpretations


def interpret_market_timing(
    market_timing_summary: pd.Series,
    significance_level: float = 0.05,
) -> list[str]:
    """Interpret the Treynor-Mazuy market timing result."""
    gamma = market_timing_summary["timing_gamma"]
    p_value = market_timing_summary["timing_p_value"]

    if not _is_significant(
        p_value,
        significance_level,
    ):
        return [
            f"The Treynor-Mazuy timing coefficient is "
            f"{gamma:.3f}, but it is not statistically "
            f"significant (p-value: {p_value:.4f}). "
            f"There is insufficient evidence of market "
            f"timing ability."
        ]

    if gamma > 0:
        return [
            f"The Treynor-Mazuy timing coefficient is "
            f"positive and statistically significant "
            f"({gamma:.3f}, p-value: {p_value:.4f}). "
            f"This provides evidence of positive market "
            f"timing ability."
        ]

    return [
        f"The Treynor-Mazuy timing coefficient is "
        f"negative and statistically significant "
        f"({gamma:.3f}, p-value: {p_value:.4f}). "
        f"This indicates adverse market timing."
    ]


def interpret_performance(
    metrics: pd.DataFrame,
    benchmark_metrics: pd.Series,
) -> list[str]:
    """Interpret the main performance and risk indicators."""
    fund = metrics.loc["fund"]
    benchmark = metrics.loc["benchmark"]

    interpretations = [
        (
            f"The fund's annualized return is "
            f"{fund['annualized_return']:.2%}, compared with "
            f"{benchmark['annualized_return']:.2%} for the "
            f"benchmark."
        ),
        (
            f"The fund's annualized volatility is "
            f"{fund['annualized_volatility']:.2%}, compared "
            f"with {benchmark['annualized_volatility']:.2%} "
            f"for the benchmark."
        ),
        (
            f"The fund's maximum drawdown is "
            f"{fund['maximum_drawdown']:.2%}, compared with "
            f"{benchmark['maximum_drawdown']:.2%} for the "
            f"benchmark."
        ),
        (
            f"The fund's beta is "
            f"{benchmark_metrics['beta']:.3f}, with a "
            f"correlation of "
            f"{benchmark_metrics['correlation']:.3f} "
            f"relative to the benchmark."
        ),
        (
            f"The information ratio is "
            f"{benchmark_metrics['information_ratio']:.3f}, "
            f"and the annualized tracking error is "
            f"{benchmark_metrics['tracking_error']:.2%}."
        ),
    ]

    return interpretations


def generate_analysis_report(
    metrics: pd.DataFrame,
    benchmark_metrics: pd.Series,
    coefficient_table: pd.DataFrame,
    model_summary: pd.Series,
    market_timing_summary: pd.Series,
    significance_level: float = 0.05,
) -> str:
    """Generate the complete automatic interpretation."""
    sections = [
        (
            "PERFORMANCE AND RISK",
            interpret_performance(
                metrics,
                benchmark_metrics,
            ),
        ),
        (
            "CARHART FOUR-FACTOR MODEL",
            interpret_carhart_model(
                coefficient_table,
                model_summary,
                significance_level,
            ),
        ),
        (
            "TREYNOR-MAZUY MARKET TIMING",
            interpret_market_timing(
                market_timing_summary,
                significance_level,
            ),
        ),
    ]

    report_lines = [
        "AUTOMATIC FUND ANALYSIS",
        "=" * 60,
    ]

    for title, paragraphs in sections:
        report_lines.extend(
            [
                "",
                title,
                "-" * len(title),
            ]
        )

        for paragraph in paragraphs:
            report_lines.append(f"- {paragraph}")

    report_lines.extend(
        [
            "",
            "IMPORTANT LIMITATION",
            "--------------------",
            (
                "Statistical significance does not guarantee "
                "future performance. Results depend on the "
                "selected period, data quality, benchmark, "
                "factor universe, and model assumptions."
            ),
        ]
    )

    return "\n".join(report_lines)