"""Chart generation for the fund analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


def _prepare_output_path(
    output_path: Path,
) -> None:
    """Create the chart output directory."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_growth_chart(
    growth_index: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the value of an initial investment."""
    _prepare_output_path(output_path)

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for column in growth_index.columns:
        axis.plot(
            growth_index.index,
            growth_index[column],
            label=column.capitalize(),
            linewidth=2,
        )

    axis.set_title(
        "Growth of an Initial Investment of 100"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio Value")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_cumulative_return_chart(
    cumulative_returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot cumulative fund and benchmark returns."""
    _prepare_output_path(output_path)

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for column in cumulative_returns.columns:
        axis.plot(
            cumulative_returns.index,
            cumulative_returns[column],
            label=column.capitalize(),
            linewidth=2,
        )

    axis.axhline(
        y=0,
        color="black",
        linewidth=0.8,
    )
    axis.set_title("Cumulative Returns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative Return")
    axis.yaxis.set_major_formatter(
        PercentFormatter(xmax=1)
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_drawdown_chart(
    daily_returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot drawdowns from previous portfolio peaks."""
    _prepare_output_path(output_path)

    wealth = (1 + daily_returns).cumprod()
    previous_peaks = wealth.cummax()
    drawdowns = wealth / previous_peaks - 1

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for column in drawdowns.columns:
        axis.plot(
            drawdowns.index,
            drawdowns[column],
            label=column.capitalize(),
            linewidth=1.8,
        )

    axis.axhline(
        y=0,
        color="black",
        linewidth=0.8,
    )
    axis.set_title("Drawdown from Previous Peak")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.yaxis.set_major_formatter(
        PercentFormatter(xmax=1)
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_factor_exposure_chart(
    coefficient_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot Carhart factor coefficients and confidence intervals."""
    _prepare_output_path(output_path)

    factor_names = [
        "MKT_RF",
        "SMB",
        "HML",
        "MOM",
    ]

    factor_data = coefficient_table.loc[
        factor_names
    ]

    coefficients = factor_data["coefficient"]
    lower_errors = (
        coefficients
        - factor_data["confidence_lower"]
    )
    upper_errors = (
        factor_data["confidence_upper"]
        - coefficients
    )

    colors = [
        "#2E86AB" if value >= 0 else "#C44536"
        for value in coefficients
    ]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.bar(
        factor_names,
        coefficients,
        color=colors,
        alpha=0.85,
        yerr=[lower_errors, upper_errors],
        capsize=6,
    )

    axis.axhline(
        y=0,
        color="black",
        linewidth=0.8,
    )
    axis.set_title(
        "Carhart Factor Exposures with 95% Confidence Intervals"
    )
    axis.set_xlabel("Factor")
    axis.set_ylabel("Estimated Coefficient")
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)