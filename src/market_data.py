"""Retrieval and preparation of Carhart factors."""

import pandas as pd
from pandas_datareader import data as web


def download_carhart_factors(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download the monthly US Carhart factors.

    Kenneth French's data are expressed as percentages.
    They are converted to decimal returns.
    """
    fama_french = web.DataReader(
        "F-F_Research_Data_Factors",
        "famafrench",
        start=start_date,
        end=end_date,
    )[0]

    momentum = web.DataReader(
        "F-F_Momentum_Factor",
        "famafrench",
        start=start_date,
        end=end_date,
    )[0]

    fama_french.columns = [
        str(column).strip()
        for column in fama_french.columns
    ]
    momentum.columns = [
        str(column).strip()
        for column in momentum.columns
    ]

    factors = fama_french.join(
        momentum,
        how="inner",
    )

    factors = factors.rename(
        columns={
            "Mkt-RF": "MKT_RF",
            "Mom": "MOM",
        }
    )

    required_columns = [
        "MKT_RF",
        "SMB",
        "HML",
        "MOM",
        "RF",
    ]

    missing_columns = set(required_columns).difference(
        factors.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing factors: {sorted(missing_columns)}"
        )

    factors = factors[required_columns]

    # Convert percentage values to decimal returns.
    factors = factors / 100.0

    factors.index.name = "month"

    return factors


def prepare_carhart_dataset(
    monthly_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align monthly returns with the Carhart factors and
    calculate the fund's excess return.
    """
    required_returns = {"fund", "benchmark"}

    if not required_returns.issubset(
        monthly_returns.columns
    ):
        raise ValueError(
            "The fund and benchmark columns are required."
        )

    returns = monthly_returns.copy()

    # The factor data use a monthly PeriodIndex.
    returns.index = returns.index.to_period("M")
    returns.index.name = "month"

    dataset = returns.join(
        factors,
        how="inner",
    ).dropna()

    dataset["fund_excess"] = (
        dataset["fund"] - dataset["RF"]
    )

    if dataset.empty:
        raise ValueError(
            "No common months were found between the fund "
            "returns and the Carhart factors."
        )

    if len(dataset) < 24:
        raise ValueError(
            "Fewer than 24 monthly observations are "
            "available, making the regression unreliable."
        )

    ordered_columns = [
        "fund",
        "benchmark",
        "RF",
        "fund_excess",
        "MKT_RF",
        "SMB",
        "HML",
        "MOM",
    ]

    return dataset[ordered_columns]