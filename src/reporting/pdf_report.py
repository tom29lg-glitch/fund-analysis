"""PDF report generation for the fund analysis."""

from html import escape
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as PdfImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DARK_BLUE = colors.HexColor("#17365D")
MEDIUM_BLUE = colors.HexColor("#4472C4")
LIGHT_BLUE = colors.HexColor("#D9EAF7")
LIGHT_GRAY = colors.HexColor("#F2F2F2")


def format_percentage(value: float) -> str:
    """Format a decimal value as a percentage."""
    return f"{float(value):.2%}"


def format_number(value: float) -> str:
    """Format a numeric value."""
    return f"{float(value):.3f}"


def add_page_number(
    canvas,
    document,
) -> None:
    """Add a footer and page number to every page."""
    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        8,
    )
    canvas.setFillColor(colors.grey)

    canvas.drawString(
        2 * cm,
        1.2 * cm,
        "Fund Financial Analysis",
    )

    canvas.drawRightString(
        A4[0] - 2 * cm,
        1.2 * cm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def create_financial_table(
    data: list[list[str]],
    column_widths: list[float],
) -> Table:
    """Create a consistently styled financial table."""
    table = Table(
        data,
        colWidths=column_widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    MEDIUM_BLUE,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "RIGHT",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, LIGHT_GRAY],
                ),
                (
                    "LINEBELOW",
                    (0, 0),
                    (-1, 0),
                    1,
                    DARK_BLUE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.25,
                    colors.lightgrey,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


def create_pdf_report(
    output_path: Path,
    fund_ticker: str,
    benchmark_ticker: str,
    start_date: str,
    end_date: str,
    metrics: pd.DataFrame,
    benchmark_metrics: pd.Series,
    coefficient_table: pd.DataFrame,
    model_summary: pd.Series,
    market_timing_table: pd.DataFrame,
    market_timing_summary: pd.Series,
    automatic_report: str,
    chart_paths: list[Path],
) -> None:
    """Create the complete fund analysis PDF report."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=2.0 * cm,
        title="Fund Financial Analysis",
        author="Python Financial Analysis Workflow",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=DARK_BLUE,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=DARK_BLUE,
            spaceBefore=10,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            spaceAfter=6,
        )
    )

    story = []

    story.append(
        Paragraph(
            "Fund Financial Analysis",
            styles["ReportTitle"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Fund:</b> {escape(fund_ticker)}<br/>"
            f"<b>Benchmark:</b> {escape(benchmark_ticker)}<br/>"
            f"<b>Analysis period:</b> "
            f"{escape(start_date)} to {escape(end_date)}<br/>"
            f"<b>Factor model:</b> Carhart Four-Factor Model",
            styles["ReportBody"],
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    story.append(
        Paragraph(
            "Executive Summary",
            styles["SectionTitle"],
        )
    )

    summary_data = [
        ["Metric", "Fund", "Benchmark"],
        [
            "Annualized return",
            format_percentage(
                metrics.loc[
                    "fund",
                    "annualized_return",
                ]
            ),
            format_percentage(
                metrics.loc[
                    "benchmark",
                    "annualized_return",
                ]
            ),
        ],
        [
            "Annualized volatility",
            format_percentage(
                metrics.loc[
                    "fund",
                    "annualized_volatility",
                ]
            ),
            format_percentage(
                metrics.loc[
                    "benchmark",
                    "annualized_volatility",
                ]
            ),
        ],
        [
            "Maximum drawdown",
            format_percentage(
                metrics.loc[
                    "fund",
                    "maximum_drawdown",
                ]
            ),
            format_percentage(
                metrics.loc[
                    "benchmark",
                    "maximum_drawdown",
                ]
            ),
        ],
        [
            "Sharpe ratio",
            format_number(
                metrics.loc[
                    "fund",
                    "sharpe_ratio",
                ]
            ),
            format_number(
                metrics.loc[
                    "benchmark",
                    "sharpe_ratio",
                ]
            ),
        ],
        [
            "Sortino ratio",
            format_number(
                metrics.loc[
                    "fund",
                    "sortino_ratio",
                ]
            ),
            format_number(
                metrics.loc[
                    "benchmark",
                    "sortino_ratio",
                ]
            ),
        ],
    ]

    story.append(
        create_financial_table(
            summary_data,
            [7.2 * cm, 4.3 * cm, 4.3 * cm],
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    benchmark_data = [
        ["Benchmark-relative metric", "Value"],
        [
            "Beta",
            format_number(benchmark_metrics["beta"]),
        ],
        [
            "Correlation",
            format_number(
                benchmark_metrics["correlation"]
            ),
        ],
        [
            "Tracking error",
            format_percentage(
                benchmark_metrics["tracking_error"]
            ),
        ],
        [
            "Information ratio",
            format_number(
                benchmark_metrics["information_ratio"]
            ),
        ],
        [
            "Annualized relative performance",
            format_percentage(
                benchmark_metrics[
                    "annualized_active_return"
                ]
            ),
        ],
    ]

    story.append(
        create_financial_table(
            benchmark_data,
            [10.5 * cm, 5.3 * cm],
        )
    )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Carhart Four-Factor Regression",
            styles["SectionTitle"],
        )
    )

    carhart_data = [
        [
            "Factor",
            "Coefficient",
            "Std. Error",
            "p-value",
        ]
    ]

    for factor, row in coefficient_table.iterrows():
        carhart_data.append(
            [
                str(factor),
                format_number(row["coefficient"]),
                format_number(row["standard_error"]),
                f"{row['p_value']:.4f}",
            ]
        )

    story.append(
        create_financial_table(
            carhart_data,
            [
                4.5 * cm,
                3.8 * cm,
                3.8 * cm,
                3.7 * cm,
            ],
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    carhart_summary_data = [
        ["Model diagnostic", "Value"],
        [
            "Observations",
            str(int(model_summary["observations"])),
        ],
        [
            "R-squared",
            format_percentage(
                model_summary["r_squared"]
            ),
        ],
        [
            "Adjusted R-squared",
            format_percentage(
                model_summary["adjusted_r_squared"]
            ),
        ],
        [
            "Monthly alpha",
            format_percentage(
                model_summary["monthly_alpha"]
            ),
        ],
        [
            "Annualized alpha",
            format_percentage(
                model_summary["annualized_alpha"]
            ),
        ],
        [
            "Durbin-Watson",
            format_number(
                model_summary["durbin_watson"]
            ),
        ],
    ]

    story.append(
        create_financial_table(
            carhart_summary_data,
            [10.5 * cm, 5.3 * cm],
        )
    )

    story.append(Spacer(1, 0.5 * cm))

    story.append(
        Paragraph(
            "Treynor-Mazuy Market Timing",
            styles["SectionTitle"],
        )
    )

    timing_data = [
        [
            "Term",
            "Coefficient",
            "Std. Error",
            "p-value",
        ]
    ]

    for term, row in market_timing_table.iterrows():
        timing_data.append(
            [
                str(term),
                format_number(row["coefficient"]),
                format_number(row["standard_error"]),
                f"{row['p_value']:.4f}",
            ]
        )

    story.append(
        create_financial_table(
            timing_data,
            [
                4.5 * cm,
                3.8 * cm,
                3.8 * cm,
                3.7 * cm,
            ],
        )
    )

    story.append(
        Paragraph(
            "The timing gamma is "
            f"{format_number(market_timing_summary['timing_gamma'])} "
            "with a p-value of "
            f"{market_timing_summary['timing_p_value']:.4f}.",
            styles["ReportBody"],
        )
    )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Automatic Interpretation",
            styles["SectionTitle"],
        )
    )

    for line in automatic_report.splitlines():
        clean_line = line.strip()

        if not clean_line:
            story.append(Spacer(1, 0.15 * cm))
            continue

        if set(clean_line) <= {"=", "-"}:
            continue

        if clean_line.isupper():
            story.append(
                Paragraph(
                    escape(clean_line),
                    styles["SectionTitle"],
                )
            )
        else:
            story.append(
                Paragraph(
                    escape(clean_line),
                    styles["ReportBody"],
                )
            )

    available_charts = [
        path
        for path in chart_paths
        if path.exists()
    ]

    for chart_path in available_charts:
        story.append(PageBreak())

        chart_title = (
            chart_path.stem
            .replace("_", " ")
            .title()
        )

        story.append(
            Paragraph(
                escape(chart_title),
                styles["SectionTitle"],
            )
        )

        image = PdfImage(
            str(chart_path),
            width=17 * cm,
            height=9.5 * cm,
        )

        story.append(image)

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )