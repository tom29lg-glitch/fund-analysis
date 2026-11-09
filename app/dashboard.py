"""Interactive Streamlit dashboard for fund analysis."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_validation import validate_and_clean_prices
from src.interpretation import generate_analysis_report
from src.market_data import (
    download_carhart_factors,
    prepare_carhart_dataset,
)
from src.metrics import (
    calculate_benchmark_metrics,
    calculate_metrics,
)
from src.models.carhart import (
    create_coefficient_table,
    create_model_summary,
    run_carhart_regression,
)
from src.models.market_timing import (
    create_market_timing_summary,
    create_market_timing_table,
    run_treynor_mazuy_regression,
)
from src.price_loader import (
    download_fund_and_benchmark,
    validate_dates,
)
from src.returns import (
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_growth_index,
    calculate_monthly_returns,
)


def run_analysis(
    fund_ticker: str,
    benchmark_ticker: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Run the complete analytical workflow."""
    validate_dates(start_date, end_date)

    raw_prices = download_fund_and_benchmark(
        fund_ticker=fund_ticker,
        benchmark_ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    clean_prices, validation_report = (
        validate_and_clean_prices(raw_prices)
    )

    daily_returns = calculate_daily_returns(
        clean_prices
    )
    monthly_returns = calculate_monthly_returns(
        clean_prices
    )
    cumulative_returns = (
        calculate_cumulative_returns(
            daily_returns
        )
    )
    growth_index = calculate_growth_index(
        daily_returns,
        initial_value=100.0,
    )

    metrics = calculate_metrics(
        daily_returns,
        annual_risk_free_rate=0.0,
        periods_per_year=252,
    )

    benchmark_metrics = (
        calculate_benchmark_metrics(
            daily_returns,
            periods_per_year=252,
        )
    )

    carhart_factors = download_carhart_factors(
        start_date=start_date,
        end_date=end_date,
    )

    carhart_dataset = prepare_carhart_dataset(
        monthly_returns=monthly_returns,
        factors=carhart_factors,
    )

    carhart_results = run_carhart_regression(
        carhart_dataset,
        hac_lags=3,
    )

    coefficient_table = create_coefficient_table(
        carhart_results,
        confidence_level=0.95,
    )

    model_summary = create_model_summary(
        carhart_results
    )

    market_timing_results = (
        run_treynor_mazuy_regression(
            carhart_dataset,
            hac_lags=3,
        )
    )

    market_timing_table = (
        create_market_timing_table(
            market_timing_results,
            confidence_level=0.95,
        )
    )

    market_timing_summary = (
        create_market_timing_summary(
            market_timing_results
        )
    )

    automatic_report = generate_analysis_report(
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        coefficient_table=coefficient_table,
        model_summary=model_summary,
        market_timing_summary=market_timing_summary,
        significance_level=0.05,
    )

    return {
        "clean_prices": clean_prices,
        "validation_report": validation_report,
        "daily_returns": daily_returns,
        "monthly_returns": monthly_returns,
        "cumulative_returns": cumulative_returns,
        "growth_index": growth_index,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
        "carhart_dataset": carhart_dataset,
        "coefficient_table": coefficient_table,
        "model_summary": model_summary,
        "market_timing_table": market_timing_table,
        "market_timing_summary": market_timing_summary,
        "automatic_report": automatic_report,
    }


def calculate_drawdowns(
    daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate drawdowns from previous peaks."""
    wealth = (1 + daily_returns).cumprod()
    previous_peaks = wealth.cummax()

    return wealth / previous_peaks - 1


def display_dashboard(
    results: dict,
) -> None:
    """Display the analysis results."""
    metrics = results["metrics"]
    benchmark_metrics = results[
        "benchmark_metrics"
    ]
    model_summary = results["model_summary"]
    timing_summary = results[
        "market_timing_summary"
    ]

    st.subheader("Key Performance Indicators")

    first_row = st.columns(4)

    first_row[0].metric(
        "Fund annualized return",
        f"{metrics.loc['fund', 'annualized_return']:.2%}",
    )
    first_row[1].metric(
        "Benchmark annualized return",
        f"{metrics.loc['benchmark', 'annualized_return']:.2%}",
    )
    first_row[2].metric(
        "Fund volatility",
        f"{metrics.loc['fund', 'annualized_volatility']:.2%}",
    )
    first_row[3].metric(
        "Maximum drawdown",
        f"{metrics.loc['fund', 'maximum_drawdown']:.2%}",
    )

    second_row = st.columns(4)

    second_row[0].metric(
        "Sharpe ratio",
        f"{metrics.loc['fund', 'sharpe_ratio']:.3f}",
    )
    second_row[1].metric(
        "Beta",
        f"{benchmark_metrics['beta']:.3f}",
    )
    second_row[2].metric(
        "Carhart alpha",
        f"{model_summary['annualized_alpha']:.2%}",
    )
    second_row[3].metric(
        "Carhart R-squared",
        f"{model_summary['r_squared']:.2%}",
    )

    tab_performance, tab_factors, tab_timing, tab_report = (
        st.tabs(
            [
                "Performance",
                "Carhart Factors",
                "Market Timing",
                "Interpretation",
            ]
        )
    )

    with tab_performance:
        st.subheader(
            "Growth of an Initial Investment of 100"
        )
        st.line_chart(
            results["growth_index"]
        )

        st.subheader("Cumulative Returns")
        st.line_chart(
            results["cumulative_returns"]
        )

        st.subheader("Drawdowns")
        drawdowns = calculate_drawdowns(
            results["daily_returns"]
        )
        st.line_chart(drawdowns)

        st.subheader("Performance Metrics")
        st.dataframe(
            metrics.style.format(
                {
                    "annualized_return": "{:.2%}",
                    "annualized_volatility": "{:.2%}",
                    "maximum_drawdown": "{:.2%}",
                    "sharpe_ratio": "{:.3f}",
                    "sortino_ratio": "{:.3f}",
                    "calmar_ratio": "{:.3f}",
                }
            ),
            width="stretch",
        )

    with tab_factors:
        st.subheader(
            "Carhart Factor Exposures"
        )

        factor_coefficients = results[
            "coefficient_table"
        ].loc[
            ["MKT_RF", "SMB", "HML", "MOM"],
            ["coefficient"],
        ]

        st.bar_chart(factor_coefficients)

        st.subheader(
            "Regression Coefficients"
        )

        st.dataframe(
            results["coefficient_table"].style.format(
                {
                    "coefficient": "{:.4f}",
                    "standard_error": "{:.4f}",
                    "t_statistic": "{:.3f}",
                    "p_value": "{:.4f}",
                    "confidence_lower": "{:.4f}",
                    "confidence_upper": "{:.4f}",
                }
            ),
            width="stretch",
        )

        st.write(
            f"Observations: "
            f"{int(model_summary['observations'])}"
        )
        st.write(
            f"Adjusted R-squared: "
            f"{model_summary['adjusted_r_squared']:.2%}"
        )
        st.write(
            f"Durbin-Watson: "
            f"{model_summary['durbin_watson']:.3f}"
        )

    with tab_timing:
        st.subheader(
            "Treynor-Mazuy Market Timing"
        )

        st.dataframe(
            results[
                "market_timing_table"
            ].style.format(
                {
                    "coefficient": "{:.4f}",
                    "standard_error": "{:.4f}",
                    "t_statistic": "{:.3f}",
                    "p_value": "{:.4f}",
                    "confidence_lower": "{:.4f}",
                    "confidence_upper": "{:.4f}",
                }
            ),
            width="stretch",
        )

        timing_columns = st.columns(3)

        timing_columns[0].metric(
            "Market beta",
            f"{timing_summary['market_beta']:.3f}",
        )
        timing_columns[1].metric(
            "Timing gamma",
            f"{timing_summary['timing_gamma']:.3f}",
        )
        timing_columns[2].metric(
            "Timing p-value",
            f"{timing_summary['timing_p_value']:.4f}",
        )

    with tab_report:
        st.subheader(
            "Automatic Interpretation"
        )
        st.text(
            results["automatic_report"]
        )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Fund Financial Analysis",
        page_icon="📊",
        layout="wide",
    )

    st.title("Fund Financial Analysis")
    st.caption(
        "Performance, risk, Carhart factors, "
        "and Treynor-Mazuy market timing"
    )

    with st.sidebar:
        st.header("Analysis Parameters")

        fund_ticker = st.text_input(
            "Fund ticker",
            value="ARKK",
        ).strip().upper()

        benchmark_ticker = st.text_input(
            "Benchmark ticker",
            value="SPY",
        ).strip().upper()

        start_value = st.date_input(
            "Start date",
            value=date(2020, 1, 1),
        )

        end_value = st.date_input(
            "End date",
            value=date(2025, 1, 1),
        )

        run_button = st.button(
            "Run Analysis",
            type="primary",
            width="stretch",
        )

        st.info(
            "The current factor dataset uses "
            "United States Carhart factors."
        )

    if not run_button:
        st.info(
            "Select the analysis parameters and "
            "click Run Analysis."
        )
        return

    if not fund_ticker or not benchmark_ticker:
        st.error(
            "Fund and benchmark tickers are required."
        )
        return

    try:
        with st.spinner(
            "Downloading data and estimating models..."
        ):
            results = run_analysis(
                fund_ticker=fund_ticker,
                benchmark_ticker=benchmark_ticker,
                start_date=start_value.isoformat(),
                end_date=end_value.isoformat(),
            )

        st.success("Analysis completed successfully.")
        display_dashboard(results)

    except Exception as error:
        st.error(f"Analysis failed: {error}")


if __name__ == "__main__":
    main()