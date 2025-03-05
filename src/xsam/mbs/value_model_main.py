import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xsam.mbs.mock_data import generate_forward_data, generate_historical_data
from xsam.mbs.value_model import (
    JointReversionModel,
    estimate_parameters_ols,
    estimate_parameters_mle,
    monte_carlo_simulation,
    stationarity_tests,
)


def plot_value_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    paths: dict[str, pd.DataFrame],
    S_OAS_inf: float,
    C_CC: float,
    simulation_dates: pd.DatetimeIndex,
) -> plt.Figure:
    """Plot historical data and Monte Carlo paths for OAS and Convexity.

    Args:
        oas_data (pd.Series): Historical data for OAS.
        cvx_data (pd.Series): Historical data for Convexity.
        paths (dict[str, pd.DataFrame]): Monte Carlo paths for OAS and Convexity.
        S_OAS_inf (float): Reversion level for OAS.
        C_CC (float): Reversion level for Convexity.
        simulation_dates (pd.DatetimeIndex): Dates for Monte Carlo simulation.

    Returns:
        plt.Figure: Figure object with the plot.
    """
    # Plot historical data and Monte Carlo paths
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot OAS
    paths_OAS = paths['oas']
    paths_OAS.index = simulation_dates
    OAS = pd.concat([oas_data, paths_OAS], axis=1).mul(1e4)
    axs[0].plot(OAS.iloc[:, 0], color='darkblue', label='OAS')
    axs[0].plot(OAS.iloc[:, 1:], color='lightblue', alpha=0.1)
    axs[0].plot(
        OAS.iloc[:, 1:].mean(axis=1),
        color='darkblue',
        linestyle=':',
        label='Projected OAS',
        alpha=1.0,
    )
    axs[0].axhline(
        y=S_OAS_inf * 1e4, color='darkblue', linestyle='--', label='Reversion Level'
    )
    axs[0].set_title('Monte Carlo Simulation of OAS')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('OAS')
    axs[0].legend()

    # Plot Convexity
    paths_C = paths['cvx']
    paths_C.index = simulation_dates
    C = pd.concat([cvx_data, paths_C], axis=1).mul(1e4)
    axs[1].plot(C.iloc[:, 0], color='darkgreen', label='Convexity')
    axs[1].plot(C.iloc[:, 1:], color='lightgreen', alpha=0.1)
    axs[1].plot(
        C.iloc[:, 1:].mean(axis=1),
        color='darkgreen',
        linestyle=':',
        label='Projected Convexity',
    )
    axs[1].axhline(
        y=C_CC * 1e4, color='darkgreen', linestyle='--', label='Reversion Level'
    )
    axs[1].set_title('Monte Carlo Simulation of Convexity')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Convexity')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return fig


def run_value_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    S_OAS_inf: float,
    C_CC: float,
    enable_convexity: bool,
    enable_volatility: bool,
    S_OAS_init: float,
    C_init: float,
    sigma_r_forward: float | pd.Series,
    nu_r_forward: float | pd.Series,
    simulation_dates: pd.DatetimeIndex,
    num_paths: int,
    seed: int,
    verbose: bool = False,
    plot: bool = True,
):
    """Run the MBS valuation model. The function performs the following steps:
    1. Perform stationarity tests for OAS and Convexity.
    2. Estimate model parameters using OLS to generate initial guess for MLE.
    3. Estimate model parameters using MLE.
    4. Perform Monte Carlo simulation.
    5. Plot historical data and Monte Carlo paths.

    Args:
        oas_data (pd.Series): Historical data for OAS.
        cvx_data (pd.Series): Historical data for Convexity.
        sigma_r_data (pd.Series): Historical data for Rates Volatility.
        nu_r_data (pd.Series): Historical data for Rates Volatility of Volatility.
        S_OAS_inf (float): Reversion level for OAS.
        C_CC (float): Reversion level for Convexity.
        enable_convexity (bool): Flag to enable Convexity.
        enable_volatility (bool): Flag to enable Volatility.
        S_OAS_init (float): Initial value for OAS at the start of the simulation.
        C_init (float): Initial value for Convexity at the start of the simulation.
        sigma_r_forward (float | pd.Series): Forward data for Rates Volatility.
        nu_r_forward (float | pd.Series): Forward data for Rates Volatility of Volatility.
        simulation_dates (pd.DatetimeIndex): Dates for Monte Carlo simulation.
        num_paths (int): Number of Monte Carlo paths.
        seed (int): Seed for reproducibility.
    """

    # Assert that all data series have the same frequency
    assert (
        oas_data.index.freq
        == cvx_data.index.freq
        == sigma_r_data.index.freq
        == nu_r_data.index.freq
    )
    assert simulation_dates.freq == oas_data.index.freq
    if isinstance(sigma_r_forward, pd.Series):
        assert sigma_r_forward.index.freq == nu_r_forward.index.freq
    if isinstance(nu_r_forward, pd.Series):
        assert sigma_r_forward.index.freq == nu_r_forward.index.freq

    # Time step for Monte Carlo simulation
    dt = 1 / len(simulation_dates)

    # Stationarity tests
    adf_OAS, adf_C = stationarity_tests(oas_data, cvx_data, verbose=verbose)

    (
        kappa_ols,
        gamma_ols,
        sigma_O_0_ols,
        delta_ols,
        lambda_ols,
        beta_ols,
        sigma_C_ols,
    ) = estimate_parameters_ols(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        S_OAS_inf,
        C_CC,
        enable_convexity=enable_convexity,
        enable_volatility=enable_volatility,
        verbose=verbose,
    )

    # Use OLS estimates as initial guess for MLE
    initial_guess = [
        kappa_ols,
        *gamma_ols,
        sigma_O_0_ols,
        delta_ols,
        lambda_ols,
        *beta_ols,
        sigma_C_ols,
    ]

    # Estimate model parameters using MLE
    (
        kappa_mle,
        gamma_mle,
        sigma_O_0_mle,
        delta_mle,
        lambda_mle,
        beta_mle,
        sigma_C_mle,
    ) = estimate_parameters_mle(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        S_OAS_inf,
        C_CC,
        dt,
        initial_guess=initial_guess,
        enable_convexity=enable_convexity,
        enable_volatility=enable_volatility,
        verbose=verbose,
    )

    # Create model instance with MLE parameters
    model = JointReversionModel(
        kappa_mle,
        lambda_mle,
        gamma_mle,
        beta_mle,
        sigma_O_0_mle,
        delta_mle,
        sigma_C_mle,
        dt,
    )

    # Perform Monte Carlo simulation
    paths = monte_carlo_simulation(
        model,
        S_OAS_init,
        C_init,
        S_OAS_inf,
        C_CC,
        sigma_r_forward,
        nu_r_forward,
        simulation_dates,
        enable_convexity,
        enable_volatility,
        num_paths,
        seed,
    )

    # Plot historical data and Monte Carlo paths
    if plot:
        plot_value_model(oas_data, cvx_data, paths, S_OAS_inf, C_CC, simulation_dates)

    return paths


def main() -> None:
    """Main function running the MBS valuation process.

    The main function performs the following steps:
    1. Set seed for reproducibility.
    2. Define variable parameters for the model.
    3. Generate training data for ZV, OAS, Rates Vol, and Rates Vol of Vol.
    4. Generate forward data for Rates Vol and Rates Vol of Vol.
    5. Perform stationarity tests for OAS and Convexity.
    6. Estimate model parameters using OLS to generate initial guess for MLE.
    7. Estimate model parameters using MLE.
    8. Perform Monte Carlo simulation.
    9. Plot historical data and Monte Carlo paths.
    """
    # Set seed for reproducibility
    seed = 42

    # Define training data parameters
    zv_params = {'mu': 0.005, 'theta': 0.01, 'sigma': 0.002, 'X0': 0.008}
    oas_params = {'mu': 0.003, 'theta': 0.02, 'sigma': 0.001, 'X0': 0.002}
    zv_oas_rho = 0.8
    sigma_r_params = {'mu': 0.002, 'theta': 0.01, 'sigma': 0.001, 'X0': 0.002}
    nu_r_params = {'mu': 0.001, 'theta': 0.01, 'sigma': 0.001, 'X0': 0.001}

    # Mock data simulation parameters
    train_start_date = '2013-01-01'
    train_end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    historical_freq = 'B'
    historical_dates = pd.date_range(
        start=train_start_date, end=train_end_date, freq=historical_freq
    )

    # Generate training data
    zv_hist, oas_hist, sigma_r_hist, nu_r_hist = generate_historical_data(
        zv_params,
        oas_params,
        zv_oas_rho,
        sigma_r_params,
        nu_r_params,
        historical_dates,
        seed,
    )
    cvx_hist = zv_hist - oas_hist

    # Monte Carlo simulation parameters
    project_num_days = 365
    project_freq = 'W-FRI'
    simulation_dates = pd.date_range(
        start=train_end_date, periods=project_num_days, freq='D'
    )
    simulation_dates = pd.Series(index=simulation_dates).resample(project_freq).first().index
    oas_data = oas_hist.resample(project_freq).last()
    cvx_data = cvx_hist.resample(project_freq).last()
    sigma_r_data = sigma_r_hist.resample(project_freq).last()
    nu_r_data = nu_r_hist.resample(project_freq).last()
    S_OAS_init = float(oas_data.iloc[-1])
    C_init = float(cvx_data.iloc[-1])
    num_paths = 100  # Number of Monte Carlo paths

    # Update X0 for forward data
    sigma_r_params['X0'] = sigma_r_data.iloc[-1]
    nu_r_params['X0'] = nu_r_data.iloc[-1]

    # Generate forward data
    sigma_r_forward, nu_r_forward = generate_forward_data(
        sigma_r_params,
        nu_r_params,
        simulation_dates,
        seed,
    )

    # Model parameters
    S_OAS_inf = float(np.mean(oas_data))
    C_CC = 0.004
    enable_convexity = True
    enable_volatility = True

    # Run MBS valuation model
    run_value_model(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        S_OAS_inf,
        C_CC,
        enable_convexity,
        enable_volatility,
        S_OAS_init,
        C_init,
        sigma_r_forward,
        nu_r_forward,
        simulation_dates,
        num_paths,
        seed,
        verbose=True,
        plot=True,
    )


if __name__ == "__main__":
    main()
