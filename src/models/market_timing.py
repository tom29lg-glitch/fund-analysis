"""Treynor-Mazuy market timing model."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import (
    RegressionResultsWrapper,
)


def run_treynor_mazuy_regression(
    dataset: pd.DataFrame,
    hac_lags: int = 3,
) -> RegressionResultsWrapper:
    """
    Estimate the Treynor-Mazuy market timing model.

    Newey-West HAC standard errors are used.
    """
    required_columns = {
        "fund_excess",
        "MKT_RF",
    }

    missing_columns = required_columns.difference(
        dataset.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing model columns: {sorted(missing_columns)}"
        )

    model_data = dataset[
        ["fund_excess", "MKT_RF"]
    ].dropna().copy()

    if len(model_data) < 24:
        raise ValueError(
            "At least 24 monthly observations are required."
        )

    model_data["MKT_RF_squared"] = (
        model_data["MKT_RF"] ** 2
    )

    y = model_data["fund_excess"]

    x = model_data[
        ["MKT_RF", "MKT_RF_squared"]
    ]
    x = sm.add_constant(x)

    model = sm.OLS(y, x)

    results = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )

    return results


def create_market_timing_table(
    results: RegressionResultsWrapper,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Create the Treynor-Mazuy coefficient table."""
    alpha = 1 - confidence_level
    confidence_intervals = results.conf_int(
        alpha=alpha
    )

    table = pd.DataFrame(
        {
            "coefficient": results.params,
            "standard_error": results.bse,
            "t_statistic": results.tvalues,
            "p_value": results.pvalues,
            "confidence_lower": confidence_intervals[0],
            "confidence_upper": confidence_intervals[1],
        }
    )

    table = table.rename(
        index={
            "const": "alpha",
            "MKT_RF": "market_beta",
            "MKT_RF_squared": "timing_gamma",
        }
    )
    table.index.name = "term"

    return table


def create_market_timing_summary(
    results: RegressionResultsWrapper,
) -> pd.Series:
    """Create the main Treynor-Mazuy diagnostics."""
    monthly_alpha = float(results.params["const"])
    timing_gamma = float(
        results.params["MKT_RF_squared"]
    )
    timing_p_value = float(
        results.pvalues["MKT_RF_squared"]
    )

    annualized_alpha = (
        (1 + monthly_alpha) ** 12 - 1
        if monthly_alpha > -1
        else np.nan
    )

    return pd.Series(
        {
            "observations": int(results.nobs),
            "r_squared": results.rsquared,
            "adjusted_r_squared": results.rsquared_adj,
            "monthly_alpha": monthly_alpha,
            "annualized_alpha": annualized_alpha,
            "market_beta": results.params["MKT_RF"],
            "timing_gamma": timing_gamma,
            "timing_p_value": timing_p_value,
        },
        name="treynor_mazuy_model",
    )