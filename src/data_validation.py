"""Price data validation and cleaning."""

import pandas as pd


def validate_and_clean_prices(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Validate, clean, and align the fund and benchmark prices.

    Returns
    -------
    cleaned_prices:
        Clean prices aligned to common dates.
    report:
        Number of detected issues and removed observations.
    """
    required_columns = {"fund", "benchmark"}

    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if prices.empty:
        raise ValueError("The price DataFrame is empty.")

    cleaned = prices.copy()

    # Explicitly convert the index to dates.
    cleaned.index = pd.to_datetime(
        cleaned.index,
        errors="coerce",
    )

    invalid_dates = int(cleaned.index.isna().sum())
    cleaned = cleaned.loc[cleaned.index.notna()]

    # Remove duplicate dates.
    duplicate_dates = int(cleaned.index.duplicated().sum())
    cleaned = cleaned.loc[
        ~cleaned.index.duplicated(keep="last")
    ]

    # Sort chronologically.
    cleaned = cleaned.sort_index()

    # Convert columns to numeric values.
    for column in required_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    missing_values = int(
        cleaned[["fund", "benchmark"]].isna().sum().sum()
    )

    # A price less than or equal to zero is invalid.
    invalid_prices_mask = (
        cleaned[["fund", "benchmark"]] <= 0
    ).any(axis=1)
    invalid_prices = int(invalid_prices_mask.sum())

    cleaned = cleaned.loc[~invalid_prices_mask]

    rows_before_alignment = len(cleaned)

    # Keep only dates for which both assets have a price.
    cleaned = cleaned.dropna(
        subset=["fund", "benchmark"],
        how="any",
    )

    rows_removed_alignment = (
        rows_before_alignment - len(cleaned)
    )

    if cleaned.empty:
        raise ValueError(
            "No common observations remain after cleaning."
        )

    if len(cleaned) < 2:
        raise ValueError(
            "At least two observations are required to "
            "calculate returns."
        )

    report = {
        "invalid_dates": invalid_dates,
        "duplicate_dates": duplicate_dates,
        "missing_values": missing_values,
        "invalid_prices": invalid_prices,
        "rows_removed_alignment": rows_removed_alignment,
        "final_rows": len(cleaned),
    }

    return cleaned, report


def display_validation_report(
    report: dict[str, int],
) -> None:
    """Display a readable summary of the cleaning process."""
    print("\n=== Validation Report ===")
    print(f"Invalid dates              : {report['invalid_dates']}")
    print(f"Duplicate dates            : {report['duplicate_dates']}")
    print(f"Missing values detected    : {report['missing_values']}")
    print(f"Zero or negative prices    : {report['invalid_prices']}")
    print(
        "Rows removed during alignment: "
        f"{report['rows_removed_alignment']}"
    )
    print(f"Final observations         : {report['final_rows']}")