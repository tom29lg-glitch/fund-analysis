"""Excel report generation for the fund analysis."""

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.worksheet import Worksheet


DARK_BLUE = "17365D"
MEDIUM_BLUE = "4472C4"
LIGHT_BLUE = "D9EAF7"
LIGHT_GREEN = "E2F0D9"
LIGHT_RED = "FCE4D6"
WHITE = "FFFFFF"
BLACK = "000000"

PERCENT_FORMAT = "0.00%;[Red](0.00%);-"
NUMBER_FORMAT = "0.000"
DATE_FORMAT = "yyyy-mm-dd"

THIN_GRAY_BORDER = Border(
    bottom=Side(
        style="thin",
        color="D9E1F2",
    )
)


def style_title(
    worksheet: Worksheet,
    title: str,
    end_column: int = 6,
) -> None:
    """Create a formatted worksheet title."""
    worksheet.merge_cells(
        start_row=1,
        start_column=1,
        end_row=1,
        end_column=end_column,
    )

    cell = worksheet.cell(row=1, column=1)
    cell.value = title
    cell.font = Font(
        color=WHITE,
        bold=True,
        size=16,
    )
    cell.fill = PatternFill(
        fill_type="solid",
        fgColor=DARK_BLUE,
    )
    cell.alignment = Alignment(
        horizontal="left",
        vertical="center",
    )

    worksheet.row_dimensions[1].height = 28
    worksheet.sheet_view.showGridLines = False


def style_header_row(
    worksheet: Worksheet,
    row_number: int,
) -> None:
    """Format a table header row."""
    for cell in worksheet[row_number]:
        if cell.value is None:
            continue

        cell.font = Font(
            color=WHITE,
            bold=True,
        )
        cell.fill = PatternFill(
            fill_type="solid",
            fgColor=MEDIUM_BLUE,
        )
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )


def adjust_column_widths(
    worksheet: Worksheet,
    maximum_width: int = 40,
) -> None:
    """Adjust column widths without creating oversized columns."""
    for column_cells in worksheet.columns:
        column_letter = get_column_letter(
            column_cells[0].column
        )

        content_lengths = [
            len(str(cell.value))
            for cell in column_cells
            if cell.value is not None
        ]

        if not content_lengths:
            continue

        width = min(
            max(content_lengths) + 2,
            maximum_width,
        )
        worksheet.column_dimensions[
            column_letter
        ].width = width


def write_dataframe_sheet(
    workbook: Workbook,
    sheet_name: str,
    title: str,
    dataframe: pd.DataFrame,
    percentage_columns: list[str] | None = None,
) -> Worksheet:
    """Write and format a DataFrame in a worksheet."""
    worksheet = workbook.create_sheet(sheet_name)
    style_title(
        worksheet,
        title,
        end_column=max(len(dataframe.columns) + 1, 4),
    )

    export_data = dataframe.copy()
    export_data.index.name = (
        export_data.index.name or "index"
    )
    export_data = export_data.reset_index()

    # Convert pandas Period values into Excel-compatible dates.
    for column in export_data.columns:
        if isinstance(
            export_data[column].dtype,
            pd.PeriodDtype,
        ):
            export_data[column] = (
                export_data[column]
                .dt.to_timestamp(how="end")
                .dt.normalize()
            )

    for row in dataframe_to_rows(
        export_data,
        index=False,
        header=True,
    ):
        worksheet.append(row)

    header_row = 2
    style_header_row(worksheet, header_row)

    worksheet.freeze_panes = "A3"
    worksheet.auto_filter.ref = worksheet.dimensions

    percentage_columns = percentage_columns or []

    column_positions = {
        cell.value: cell.column
        for cell in worksheet[header_row]
    }

    for column_name in percentage_columns:
        column_number = column_positions.get(column_name)

        if column_number is None:
            continue

        for row_number in range(
            header_row + 1,
            worksheet.max_row + 1,
        ):
            worksheet.cell(
                row=row_number,
                column=column_number,
            ).number_format = PERCENT_FORMAT

    for row in worksheet.iter_rows(
        min_row=header_row + 1
    ):
        for cell in row:
            if isinstance(
                cell.value,
                pd.Timestamp,
            ):
                cell.number_format = DATE_FORMAT

    adjust_column_widths(worksheet)

    return worksheet


def create_summary_sheet(
    workbook: Workbook,
    fund_ticker: str,
    benchmark_ticker: str,
    start_date: str,
    end_date: str,
    metrics: pd.DataFrame,
    benchmark_metrics: pd.Series,
    model_summary: pd.Series,
    market_timing_summary: pd.Series,
) -> Worksheet:
    """Create the main financial summary sheet."""
    worksheet = workbook.active
    worksheet.title = "Summary"

    style_title(
        worksheet,
        "Fund Financial Analysis",
        end_column=6,
    )

    metadata = [
        ("Fund", fund_ticker),
        ("Benchmark", benchmark_ticker),
        ("Start date", start_date),
        ("End date", end_date),
        ("Model", "Carhart Four-Factor Model"),
    ]

    worksheet["A3"] = "Analysis Information"
    worksheet["A3"].font = Font(
        bold=True,
        color=WHITE,
    )
    worksheet["A3"].fill = PatternFill(
        fill_type="solid",
        fgColor=MEDIUM_BLUE,
    )

    for row_number, (label, value) in enumerate(
        metadata,
        start=4,
    ):
        worksheet.cell(
            row=row_number,
            column=1,
            value=label,
        ).font = Font(bold=True)

        worksheet.cell(
            row=row_number,
            column=2,
            value=value,
        )

    worksheet["A11"] = "Key Performance Indicators"
    worksheet["A11"].font = Font(
        bold=True,
        color=WHITE,
    )
    worksheet["A11"].fill = PatternFill(
        fill_type="solid",
        fgColor=MEDIUM_BLUE,
    )

    kpis = [
        (
            "Fund annualized return",
            metrics.loc[
                "fund",
                "annualized_return",
            ],
            PERCENT_FORMAT,
        ),
        (
            "Benchmark annualized return",
            metrics.loc[
                "benchmark",
                "annualized_return",
            ],
            PERCENT_FORMAT,
        ),
        (
            "Fund annualized volatility",
            metrics.loc[
                "fund",
                "annualized_volatility",
            ],
            PERCENT_FORMAT,
        ),
        (
            "Fund maximum drawdown",
            metrics.loc[
                "fund",
                "maximum_drawdown",
            ],
            PERCENT_FORMAT,
        ),
        (
            "Fund Sharpe ratio",
            metrics.loc[
                "fund",
                "sharpe_ratio",
            ],
            NUMBER_FORMAT,
        ),
        (
            "Fund beta",
            benchmark_metrics["beta"],
            NUMBER_FORMAT,
        ),
        (
            "Information ratio",
            benchmark_metrics[
                "information_ratio"
            ],
            NUMBER_FORMAT,
        ),
        (
            "Carhart annualized alpha",
            model_summary["annualized_alpha"],
            PERCENT_FORMAT,
        ),
        (
            "Carhart R-squared",
            model_summary["r_squared"],
            PERCENT_FORMAT,
        ),
        (
            "Market timing gamma",
            market_timing_summary["timing_gamma"],
            NUMBER_FORMAT,
        ),
        (
            "Market timing p-value",
            market_timing_summary["timing_p_value"],
            NUMBER_FORMAT,
        ),
    ]

    for row_number, (
        label,
        value,
        number_format,
    ) in enumerate(kpis, start=12):
        label_cell = worksheet.cell(
            row=row_number,
            column=1,
            value=label,
        )
        value_cell = worksheet.cell(
            row=row_number,
            column=2,
            value=float(value),
        )

        label_cell.border = THIN_GRAY_BORDER
        value_cell.border = THIN_GRAY_BORDER
        value_cell.number_format = number_format
        value_cell.alignment = Alignment(
            horizontal="right"
        )

    worksheet.column_dimensions["A"].width = 34
    worksheet.column_dimensions["B"].width = 22
    worksheet.sheet_view.showGridLines = False

    return worksheet


def add_chart_images(
    worksheet: Worksheet,
    chart_paths: list[Path],
) -> None:
    """Add available PNG charts to a worksheet."""
    positions = ["A3", "J3", "A25", "J25"]

    for chart_path, position in zip(
        chart_paths,
        positions,
    ):
        if not chart_path.exists():
            continue

        image = Image(chart_path)
        image.width = 620
        image.height = 340
        worksheet.add_image(image, position)


def create_excel_report(
    output_path: Path,
    fund_ticker: str,
    benchmark_ticker: str,
    start_date: str,
    end_date: str,
    clean_prices: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    benchmark_metrics: pd.Series,
    carhart_dataset: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    model_summary: pd.Series,
    market_timing_table: pd.DataFrame,
    market_timing_summary: pd.Series,
    automatic_report: str,
    chart_paths: list[Path],
) -> None:
    """Create the complete Excel analysis report."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    workbook = Workbook()

    create_summary_sheet(
        workbook=workbook,
        fund_ticker=fund_ticker,
        benchmark_ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        model_summary=model_summary,
        market_timing_summary=market_timing_summary,
    )

    write_dataframe_sheet(
        workbook,
        "Performance Metrics",
        "Performance and Risk Metrics",
        metrics,
        percentage_columns=[
            "annualized_return",
            "annualized_volatility",
            "maximum_drawdown",
        ],
    )

    write_dataframe_sheet(
        workbook,
        "Benchmark Analysis",
        "Fund Versus Benchmark",
        benchmark_metrics.to_frame("value"),
        percentage_columns=["value"],
    )

    write_dataframe_sheet(
        workbook,
        "Carhart Regression",
        "Carhart Four-Factor Regression",
        coefficient_table,
    )

    write_dataframe_sheet(
        workbook,
        "Carhart Summary",
        "Carhart Model Diagnostics",
        model_summary.to_frame("value"),
    )

    write_dataframe_sheet(
        workbook,
        "Market Timing",
        "Treynor-Mazuy Market Timing",
        market_timing_table,
    )

    write_dataframe_sheet(
        workbook,
        "Monthly Returns",
        "Monthly Fund and Benchmark Returns",
        monthly_returns,
        percentage_columns=["fund", "benchmark"],
    )

    write_dataframe_sheet(
        workbook,
        "Model Dataset",
        "Carhart Model Dataset",
        carhart_dataset,
        percentage_columns=[
            "fund",
            "benchmark",
            "RF",
            "fund_excess",
            "MKT_RF",
            "SMB",
            "HML",
            "MOM",
        ],
    )

    write_dataframe_sheet(
        workbook,
        "Clean Prices",
        "Clean Daily Prices",
        clean_prices,
    )

    interpretation_sheet = workbook.create_sheet(
        "Interpretation"
    )
    style_title(
        interpretation_sheet,
        "Automatic Interpretation",
        end_column=8,
    )
    interpretation_sheet["A3"] = automatic_report
    interpretation_sheet["A3"].alignment = Alignment(
        wrap_text=True,
        vertical="top",
    )
    interpretation_sheet.column_dimensions["A"].width = 110
    interpretation_sheet.row_dimensions[3].height = 500
    interpretation_sheet.sheet_view.showGridLines = False

    charts_sheet = workbook.create_sheet("Charts")
    style_title(
        charts_sheet,
        "Fund Analysis Charts",
        end_column=18,
    )
    add_chart_images(
        charts_sheet,
        chart_paths,
    )
    charts_sheet.sheet_view.showGridLines = False

    workbook.save(output_path)