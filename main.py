"""Entry point for the fund financial analysis."""

from pathlib import Path
from src.reporting.pdf_report import create_pdf_report
from src.reporting.excel_report import create_excel_report

from src.reporting.charts import (
    create_cumulative_return_chart,
    create_drawdown_chart,
    create_factor_exposure_chart,
    create_growth_chart,
)
from src.data_validation import (
    display_validation_report,
    validate_and_clean_prices,
)
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


def build_output_paths() -> dict[str, Path]:
    """Build all data and report output paths."""
    return {
        "raw_prices": Path("data/raw/prices.csv"),
        "clean_prices": Path("data/processed/clean_prices.csv"),
        "daily_returns": Path("data/processed/daily_returns.csv"),
        "monthly_returns": Path("data/processed/monthly_returns.csv"),
        "cumulative_returns": Path(
            "data/processed/cumulative_returns.csv"
        ),
        "growth_index": Path("data/processed/growth_index.csv"),
        "performance_metrics": Path(
            "data/processed/performance_metrics.csv"
        ),
        "benchmark_metrics": Path(
            "data/processed/benchmark_metrics.csv"
        ),
        "carhart_factors": Path("data/cache/carhart_factors.csv"),
        "carhart_dataset": Path(
            "data/processed/carhart_dataset.csv"
        ),
        "carhart_coefficients": Path(
            "data/processed/carhart_coefficients.csv"
        ),
        "carhart_summary": Path(
            "data/processed/carhart_model_summary.csv"
        ),
        "carhart_report": Path(
            "outputs/reports/carhart_regression.txt"
        ),
        "market_timing_coefficients": Path(
            "data/processed/treynor_mazuy_coefficients.csv"
        ),
        "market_timing_summary": Path(
            "data/processed/treynor_mazuy_summary.csv"
        ),
        "market_timing_report": Path(
            "outputs/reports/treynor_mazuy_regression.txt"
        ),
        "automatic_interpretation": Path(
            "outputs/reports/automatic_interpretation.txt"
        ),
                "growth_chart": Path(
            "outputs/charts/growth_index.png"
        ),
        "cumulative_return_chart": Path(
            "outputs/charts/cumulative_returns.png"
        ),
        "drawdown_chart": Path(
            "outputs/charts/drawdown.png"
        ),
        "factor_exposure_chart": Path(
            "outputs/charts/factor_exposures.png"
        ),
                "excel_report": Path(
            "outputs/reports/fund_analysis.xlsx"
        ),
                "pdf_report": Path(
            "outputs/reports/fund_analysis.pdf"
        ),
    }


def create_output_directories(paths: dict[str, Path]) -> None:
    """Create the parent directory for every output file."""
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def display_results(
    validation_report: dict[str, int],
    clean_prices,
    daily_returns,
    monthly_returns,
    cumulative_returns,
    growth_index,
    metrics,
    benchmark_metrics,
    carhart_factors,
    carhart_dataset,
    coefficient_table,
    model_summary,
    market_timing_table,
    market_timing_summary,
    automatic_report: str,
    paths: dict[str, Path],
) -> None:
    """Display the main analysis results in the terminal."""
    print("\nDownload completed.")
    print(f"Raw data: {paths['raw_prices']}")

    display_validation_report(validation_report)

    print(
        "\nCleaned period: "
        f"{clean_prices.index.min().date()} "
        f"→ {clean_prices.index.max().date()}"
    )
    print(f"Cleaned data: {paths['clean_prices']}")
    print("\nCleaned price preview:")
    print(clean_prices.head())

    print("\n=== Returns ===")
    print(f"Daily observations  : {len(daily_returns)}")
    print(f"Monthly observations: {len(monthly_returns)}")
    print("\nLatest monthly returns:")
    print(monthly_returns.tail())

    print("\nFinal cumulative return:")
    for column in cumulative_returns.columns:
        print(
            f"{column:10s}: "
            f"{cumulative_returns[column].iloc[-1]:.2%}"
        )

    print("\nFinal value of an initial investment of 100:")
    for column in growth_index.columns:
        print(f"{column:10s}: {growth_index[column].iloc[-1]:,.2f}")

    print("\n=== Annualized Metrics ===")
    display_metrics = metrics.copy()
    percentage_columns = [
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
    ]
    display_metrics[percentage_columns] *= 100
    print(display_metrics.round(3))
    print(
        "\nReturn, volatility, and drawdown are displayed "
        "as percentages."
    )
    print("Sharpe, Sortino, and Calmar ratios are dimensionless.")

    print("\n=== Fund Versus Benchmark ===")
    print(f"Beta                     : {benchmark_metrics['beta']:.3f}")
    print(
        f"Correlation              : "
        f"{benchmark_metrics['correlation']:.3f}"
    )
    print(
        f"Annualized tracking error: "
        f"{benchmark_metrics['tracking_error']:.2%}"
    )
    print(
        f"Information ratio        : "
        f"{benchmark_metrics['information_ratio']:.3f}"
    )
    print(
        f"Relative performance     : "
        f"{benchmark_metrics['annualized_active_return']:.2%}"
    )

    print("\n=== Carhart Factors ===")
    print(f"Available months   : {len(carhart_factors)}")
    print(f"Common months used : {len(carhart_dataset)}")
    print("\nLatest observations in the model dataset:")
    print(carhart_dataset.tail())
    print(f"\nFactors saved to     : {paths['carhart_factors']}")
    print(f"Model dataset saved to: {paths['carhart_dataset']}")

    print("\n=== Carhart Four-Factor Regression ===")
    print(
        coefficient_table[
            ["coefficient", "standard_error", "p_value"]
        ].round(4)
    )
    print("\n=== Model Diagnostics ===")
    print(f"Observations       : {int(model_summary['observations'])}")
    print(f"R-squared          : {model_summary['r_squared']:.3f}")
    print(
        f"Adjusted R-squared : "
        f"{model_summary['adjusted_r_squared']:.3f}"
    )
    print(f"Monthly alpha      : {model_summary['monthly_alpha']:.2%}")
    print(
        f"Annualized alpha   : "
        f"{model_summary['annualized_alpha']:.2%}"
    )
    print(f"Durbin-Watson      : {model_summary['durbin_watson']:.3f}")
    print(f"\nRegression report saved to: {paths['carhart_report']}")

    print("\n=== Treynor-Mazuy Market Timing ===")
    print(
        market_timing_table[
            ["coefficient", "standard_error", "p_value"]
        ].round(4)
    )
    print("\n=== Market Timing Diagnostics ===")
    print(f"R-squared      : {market_timing_summary['r_squared']:.3f}")
    print(f"Market beta    : {market_timing_summary['market_beta']:.3f}")
    print(f"Timing gamma   : {market_timing_summary['timing_gamma']:.3f}")
    print(
        f"Timing p-value : "
        f"{market_timing_summary['timing_p_value']:.4f}"
    )
    print(
        f"\nMarket timing report saved to: "
        f"{paths['market_timing_report']}"
    )

    print(f"\n\n{automatic_report}")
    print(
        f"\nAutomatic interpretation saved to: "
        f"{paths['automatic_interpretation']}"
    )

    print("\n=== Generated Charts ===")
    print(f"Growth index       : {paths['growth_chart']}")
    print(
        f"Cumulative returns : "
        f"{paths['cumulative_return_chart']}"
    )
    print(f"Drawdown           : {paths['drawdown_chart']}")
    print(
        f"Factor exposures   : "
        f"{paths['factor_exposure_chart']}"
    )
    print("\n=== Excel Report ===")
    print(f"Excel report: {paths['excel_report']}")

    print("\n=== PDF Report ===")
    print(f"PDF report: {paths['pdf_report']}")


def main() -> None:
    """Run the complete fund analysis workflow."""
    print("\n=== Fund Financial Analysis ===")
    print("Enter the symbols used by Yahoo Finance.\n")

    fund_ticker = input("Fund ticker: ").strip()
    benchmark_ticker = input(
        "Benchmark ticker (example: SPY): "
    ).strip()
    start_date = input("Start date (YYYY-MM-DD): ").strip()
    end_date = input(
        "End date, not included (YYYY-MM-DD): "
    ).strip()

    try:
        validate_dates(start_date, end_date)

        paths = build_output_paths()
        create_output_directories(paths)

        # 1. Download, validate, and clean prices.
        raw_prices = download_fund_and_benchmark(
            fund_ticker=fund_ticker,
            benchmark_ticker=benchmark_ticker,
            start_date=start_date,
            end_date=end_date,
        )
        clean_prices, validation_report = validate_and_clean_prices(
            raw_prices
        )

        # 2. Calculate returns and growth series.
        daily_returns = calculate_daily_returns(clean_prices)
        monthly_returns = calculate_monthly_returns(clean_prices)
        cumulative_returns = calculate_cumulative_returns(daily_returns)
        growth_index = calculate_growth_index(
            daily_returns,
            initial_value=100.0,
        )

        # 3. Calculate absolute and benchmark-relative metrics.
        metrics = calculate_metrics(
            daily_returns,
            annual_risk_free_rate=0.0,
            periods_per_year=252,
        )
        benchmark_metrics = calculate_benchmark_metrics(
            daily_returns,
            periods_per_year=252,
        )

        # 4. Download factors and prepare the monthly model dataset.
        carhart_factors = download_carhart_factors(
            start_date=start_date,
            end_date=end_date,
        )
        carhart_dataset = prepare_carhart_dataset(
            monthly_returns=monthly_returns,
            factors=carhart_factors,
        )

        # 5. Estimate the Carhart four-factor model.
        carhart_results = run_carhart_regression(
            carhart_dataset,
            hac_lags=3,
        )
        coefficient_table = create_coefficient_table(
            carhart_results,
            confidence_level=0.95,
        )
        model_summary = create_model_summary(carhart_results)

        # 6. Estimate the Treynor-Mazuy market timing model.
        market_timing_results = run_treynor_mazuy_regression(
            carhart_dataset,
            hac_lags=3,
        )
        market_timing_table = create_market_timing_table(
            market_timing_results,
            confidence_level=0.95,
        )
        market_timing_summary = create_market_timing_summary(
            market_timing_results
        )

        # 7. Generate the interpretation after all inputs exist.
        automatic_report = generate_analysis_report(
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            coefficient_table=coefficient_table,
            model_summary=model_summary,
            market_timing_summary=market_timing_summary,
            significance_level=0.05,
        )

        # 8. Generate analysis charts.
        create_growth_chart(
            growth_index=growth_index,
            output_path=paths["growth_chart"],
        )

        create_cumulative_return_chart(
            cumulative_returns=cumulative_returns,
            output_path=paths["cumulative_return_chart"],
        )

        create_drawdown_chart(
            daily_returns=daily_returns,
            output_path=paths["drawdown_chart"],
        )

        create_factor_exposure_chart(
            coefficient_table=coefficient_table,
            output_path=paths["factor_exposure_chart"],
        )
        # 9. Generate the Excel report.
        create_excel_report(
            output_path=paths["excel_report"],
            fund_ticker=fund_ticker,
            benchmark_ticker=benchmark_ticker,
            start_date=start_date,
            end_date=end_date,
            clean_prices=clean_prices,
            monthly_returns=monthly_returns,
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            carhart_dataset=carhart_dataset,
            coefficient_table=coefficient_table,
            model_summary=model_summary,
            market_timing_table=market_timing_table,
            market_timing_summary=market_timing_summary,
            automatic_report=automatic_report,
            chart_paths=[
                paths["growth_chart"],
                paths["cumulative_return_chart"],
                paths["drawdown_chart"],
                paths["factor_exposure_chart"],
            ],
        )

        # 10. Generate the PDF report.
        create_pdf_report(
            output_path=paths["pdf_report"],
            fund_ticker=fund_ticker,
            benchmark_ticker=benchmark_ticker,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            coefficient_table=coefficient_table,
            model_summary=model_summary,
            market_timing_table=market_timing_table,
            market_timing_summary=market_timing_summary,
            automatic_report=automatic_report,
            chart_paths=[
                paths["growth_chart"],
                paths["cumulative_return_chart"],
                paths["drawdown_chart"],
                paths["factor_exposure_chart"],
            ],
        )

        # 11. Save all datasets and reports.
        raw_prices.to_csv(paths["raw_prices"])
        clean_prices.to_csv(paths["clean_prices"])
        daily_returns.to_csv(paths["daily_returns"])
        monthly_returns.to_csv(paths["monthly_returns"])
        cumulative_returns.to_csv(paths["cumulative_returns"])
        growth_index.to_csv(paths["growth_index"])
        metrics.to_csv(paths["performance_metrics"])
        benchmark_metrics.to_csv(
            paths["benchmark_metrics"],
            header=True,
        )
        carhart_factors.to_csv(paths["carhart_factors"])
        carhart_dataset.to_csv(paths["carhart_dataset"])
        coefficient_table.to_csv(paths["carhart_coefficients"])
        model_summary.to_csv(paths["carhart_summary"], header=True)
        market_timing_table.to_csv(
            paths["market_timing_coefficients"]
        )
        market_timing_summary.to_csv(
            paths["market_timing_summary"],
            header=True,
        )
        paths["carhart_report"].write_text(
            carhart_results.summary().as_text(),
            encoding="utf-8",
        )
        paths["market_timing_report"].write_text(
            market_timing_results.summary().as_text(),
            encoding="utf-8",
        )
        paths["automatic_interpretation"].write_text(
            automatic_report,
            encoding="utf-8",
        )

        # 12. Display the completed analysis.
        display_results(
            validation_report=validation_report,
            clean_prices=clean_prices,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            cumulative_returns=cumulative_returns,
            growth_index=growth_index,
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            carhart_factors=carhart_factors,
            carhart_dataset=carhart_dataset,
            coefficient_table=coefficient_table,
            model_summary=model_summary,
            market_timing_table=market_timing_table,
            market_timing_summary=market_timing_summary,
            automatic_report=automatic_report,
            paths=paths,
        )

    except (ValueError, KeyError) as error:
        print(f"\nValidation error: {error}")
    except Exception as error:
        print(f"\nError during the analysis: {error}")


if __name__ == "__main__":
    main()
