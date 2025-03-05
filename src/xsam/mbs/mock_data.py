import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xsam.mbs.stochastic_processes import (
    generate_mean_reversion,
    generate_correlated_mean_reversion,
)


def generate_series(
    params: dict[str, float], simulation_dates: pd.DatetimeIndex, seed: int = 42
) -> pd.Series:
    """Generate mean reversion series.

    Args:
        params (dict[str, float]): Mean reversion parameters.
        simulation_dates (pd.DatetimeIndex): Simulation dates for the series.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.Series: Mean reversion series.
    """
    np.random.seed(seed)
    T = len(simulation_dates) - 1
    N = len(simulation_dates) - 1
    t, series = generate_mean_reversion(
        params["mu"], params["theta"], params["sigma"], params["X0"], T, N, seed
    )
    return pd.Series(series, index=simulation_dates)


def generate_correlated_series(
    params1: dict[str, float],
    params2: dict[str, float],
    rho: float,
    simulation_dates: pd.DatetimeIndex,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Generate correlated mean reversion series.

    Args:
        params1 (dict[str, float]): Mean reversion parameters for series 1.
        params2 (dict[str, float]): Mean reversion parameters for series 2.
        rho (float): Correlation coefficient between series 1 and series 2.
        simulation_dates (pd.DatetimeIndex): Simulation dates for the series.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[pd.Series, pd.Series]: series1, series2
    """
    np.random.seed(seed)
    T = len(simulation_dates) - 1
    N = len(simulation_dates) - 1
    t, series1, series2 = generate_correlated_mean_reversion(
        params1["mu"],
        params1["theta"],
        params1["sigma"],
        params1["X0"],
        params2["mu"],
        params2["theta"],
        params2["sigma"],
        params2["X0"],
        T,
        N,
        rho,
        seed,
    )
    return pd.Series(series1, index=simulation_dates), pd.Series(
        series2, index=simulation_dates
    )


def generate_historical_data(
    zv_params: dict[str, float],
    oas_params: dict[str, float],
    zv_oas_rho: float,
    sigma_r_params: dict[str, float],
    nu_r_params: dict[str, float],
    simulation_dates: pd.DatetimeIndex,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate historical training data for MBS ZV, MBS OAS, Rates Vol, and Rates Vol of Vol.

    Args:
        zv_params (dict[str, float]): ZV mean reversion parameters.
        oas_params (dict[str, float]): OAS mean reversion parameters.
        zv_oas_rho (float): Correlation coefficient between ZV and OAS.
        sigma_r_params (dict[str, float]): Sigma_r mean reversion parameters.
        nu_r_params (dict[str, float]): Nu_r mean reversion parameters.
        simulation_dates (pd.DatetimeIndex): Simulation dates for historical data.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series, pd.Series]: zv_data, oas_data, sigma_r_data, nu_r_data
    """
    # Generate past training spread data
    zv_data, oas_data = generate_correlated_series(
        zv_params, oas_params, zv_oas_rho, simulation_dates, seed
    )
    sigma_r_data = generate_series(sigma_r_params, simulation_dates, seed)
    nu_r_data = generate_series(nu_r_params, simulation_dates, seed)

    return zv_data, oas_data, sigma_r_data, nu_r_data


def generate_forward_data(
    sigma_r_params: dict[str, float],
    nu_r_params: dict[str, float],
    simulation_dates: pd.DatetimeIndex,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Generate forward data for Rates Vol and Rates Vol of Vol.

    Args:
        sigma_r_params (dict[str, float]): Parameters for sigma_r mean reversion process.
        nu_r_params (dict[str, float]): Parameters for nu_r mean reversion process.
        simulation_dates (pd.DatetimeIndex): Simulation dates for forward data.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[pd.Series, pd.Series]: sigma_r_data, nu_r_data
    """

    sigma_r_data = generate_series(sigma_r_params, simulation_dates, seed)
    nu_r_data = generate_series(nu_r_params, simulation_dates, seed)

    return sigma_r_data, nu_r_data


def plot_mock_data(data: pd.DataFrame) -> None:
    data.plot()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define variable parameters
    zv_params = {"mu": 0.005, "theta": 0.01, "sigma": 0.008, "X0": 0.005}
    oas_params = {"mu": 0.003, "theta": 0.01, "sigma": 0.005, "X0": 0.003}
    zv_oas_rho = 0.8
    sigma_r_params = {"mu": 0.02, "theta": 0.01, "sigma": 0.002, "X0": 0.02}
    nu_r_params = {"mu": 0.01, "theta": 0.01, "sigma": 0.002, "X0": 0.01}

    # Define simulation parameters
    train_start_date = "2013-01-01"
    train_end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    fwd_num_days = 252
    freq = "B"
    seed = 42

    # Generate training data
    zv_data, oas_data, sigma_r_data, nu_r_data = generate_historical_data(
        zv_params,
        oas_params,
        zv_oas_rho,
        sigma_r_params,
        nu_r_params,
        train_start_date,
        train_end_date,
        freq,
        seed,
    )

    # Update X0 for forward data
    sigma_r_params["X0"] = sigma_r_data[-1]
    nu_r_params["X0"] = nu_r_data[-1]

    # Generate forward data
    sigma_r_forward, nu_r_forward = generate_forward_data(
        sigma_r_params, nu_r_params, train_end_date, fwd_num_days, freq, seed
    )

    mock_data = pd.DataFrame(
        {
            "MBS ZV": zv_data,
            "MBS OAS": oas_data,
            "Rates Vol": sigma_r_data,
            "Rates Vol of Vol": nu_r_data,
            "Rates Vol Forward": sigma_r_forward,
            "Rates Vol of Vol Forward": nu_r_forward,
        }
    ).mul(1e4)  # Scale to basis points
    plot_mock_data(mock_data)
