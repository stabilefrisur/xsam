import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xsam.constants import DATE_FORMAT
from xsam.mbs.mock_data import generate_forward_data, generate_historical_data
from xsam.mbs.value_model import JointReversionModel


def run_value_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    enable_spread_cvx: bool,
    enable_rate_vol: bool,
    enable_local_vol: bool,
    enable_mle: bool,
    estimation_freq: str,
    simulation_freq: str,
    simulation_steps: int,
    num_paths: int,
    seed: int,
    model_param_overrides: dict[str, float] = {},
    simulation_param_overrides: dict[str, float] = {},
    verbose: bool = False,
    run_adf: bool = False,
) -> tuple[dict[str, pd.DataFrame], JointReversionModel]:
    """Run the MBS valuation model. The function performs the following steps:
    1. Resample historical data to the estimation frequency.
    2. Initialise the model.
    3. Perform stationarity tests for OAS and Convexity.
    4. Estimate model parameters using OLS.
    5. Estimate model parameters using MLE if enabled.
    6. Perform Monte Carlo simulation.

    Args:
        oas_data (pd.Series): Historical data for OAS.
        cvx_data (pd.Series): Historical data for Convexity.
        sigma_r_data (pd.Series): Historical data for Rates Volatility.
        nu_r_data (pd.Series): Historical data for Rates Volatility of Volatility.
        enable_spread_cvx (bool): Flag to enable Convexity.
        enable_rate_vol (bool): Flag to enable Volatility.
        enable_local_vol (bool): Flag to enable Local Volatility.
        enable_mle (bool): Flag to enable MLE estimation.
        estimation_freq (str): Frequency for estimation.
        simulation_freq (str): Frequency for Monte Carlo simulation.
        simulation_steps (int): Number of simulation steps.
        num_paths (int): Number of Monte Carlo paths.
        seed (int): Seed for reproducibility.
        model_param_overrides (dict[str, float], optional): Model parameter overrides. Defaults to {}.
        simulation_param_overrides (dict[str, float], optional): Simulation parameter overrides. Defaults to {}.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.
        run_adf (bool, optional): Flag to enable ADF stationarity tests. Defaults to False.
    """
    # Resample historical data to the estimation frequency
    oas_data = oas_data.asfreq('D').ffill().asfreq(estimation_freq)
    cvx_data = cvx_data.asfreq('D').ffill().asfreq(estimation_freq)
    sigma_r_data = sigma_r_data.asfreq('D').ffill().asfreq(estimation_freq)
    nu_r_data = nu_r_data.asfreq('D').ffill().asfreq(estimation_freq)

    # Monte Carlo simulation parameters
    simulation_params = {
        "S_OAS_init": float(oas_data.iloc[-1]),
        "C_init": float(cvx_data.iloc[-1]),
        "S_OAS_inf": float(np.mean(oas_data)),
        "C_inf": float(np.mean(cvx_data)),
        "sigma_r_forward": float(sigma_r_data.iloc[-1]),
        "nu_r_forward": float(nu_r_data.iloc[-1]),
    }

    # Override simulation parameters if specified
    simulation_params.update(simulation_param_overrides)

    # Derive time steps for simulation in both estimation and simulation frequencies
    start_date = oas_data.index[-1]
    simulation_dates = pd.date_range(
        start=start_date, periods=simulation_steps, freq=simulation_freq
    )
    simulation_dt = 1 / simulation_steps
    simulation_dates_estimation_freq = pd.date_range(
        start=simulation_dates[0], end=simulation_dates[-1], freq=estimation_freq
    )
    simulation_steps_estimation_freq = len(simulation_dates_estimation_freq)
    simulation_dt_estimation_freq = 1 / simulation_steps_estimation_freq
    if verbose:
        print("\nEstimation config:")
        print(f"Steps: {simulation_steps_estimation_freq}")
        print(f"Frequency: '{estimation_freq}'")
        est_start = oas_data.index[0].strftime(DATE_FORMAT)
        est_end = oas_data.index[-1].strftime(DATE_FORMAT)
        print(f"Range: from {est_start} to {est_end}")
        print("\nSimulation config:")
        print(f"Steps: {simulation_steps}")
        print(f"Frequency: '{simulation_freq}'")
        sim_start = simulation_dates[0].strftime(DATE_FORMAT)
        sim_end = simulation_dates[-1].strftime(DATE_FORMAT)
        print(f"Range: from {sim_start} to {sim_end}")

    # Assert that forward data series have the simulation frequency
    if isinstance(simulation_params["sigma_r_forward"], pd.Series):
        assert simulation_params["sigma_r_forward"].index.freq == simulation_dates.freq
    if isinstance(simulation_params["nu_r_forward"], pd.Series):
        assert simulation_params["nu_r_forward"].index.freq == simulation_dates.freq

    # Initialise the model
    model = JointReversionModel(
        dt=simulation_dt,
        enable_spread_cvx=enable_spread_cvx,
        enable_rate_vol=enable_rate_vol,
        enable_local_vol=enable_local_vol,
    )

    # Stationarity tests
    if run_adf:
        adf_OAS, adf_C = model.stationarity_tests_adf(oas_data, cvx_data, verbose=verbose)

    # Estimate model parameters using OLS
    model.estimate_parameters_ols(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        simulation_params["S_OAS_inf"],
        simulation_params["C_inf"],
        simulation_steps_estimation_freq,
        verbose=verbose,
    )

    if enable_mle:
        # Estimate model parameters using MLE
        model.estimate_parameters_mle(
            oas_data,
            cvx_data,
            sigma_r_data,
            nu_r_data,
            simulation_params["S_OAS_inf"],
            simulation_params["C_inf"],
            simulation_dt_estimation_freq,
            verbose=verbose,
        )

    # Override model parameters if specified
    model.update_parameters(**model_param_overrides)

    # Perform Monte Carlo simulation
    paths = model.monte_carlo_simulation(
        **simulation_params,
        simulation_dates=simulation_dates,
        num_paths=num_paths,
        seed=seed,
        verbose=verbose,
    )

    return paths, model


def plot_value_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    paths: dict[str, pd.DataFrame],
    S_OAS_inf: float,
    C_inf: float,
) -> plt.Figure:
    """Plot historical data and Monte Carlo paths for OAS and Convexity.

    Args:
        oas_data (pd.Series): Historical data for OAS.
        cvx_data (pd.Series): Historical data for Convexity.
        paths (dict[str, pd.DataFrame]): Monte Carlo paths for OAS and Convexity.
        S_OAS_inf (float): Reversion level for OAS.
        C_inf (float): Reversion level for Convexity.

    Returns:
        plt.Figure: Figure object with the plot.
    """
    # Plot historical data and Monte Carlo paths
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axs = axs.flatten()

    # Plot OAS
    paths_OAS = paths["oas"]
    OAS = pd.concat([oas_data, paths_OAS], axis=1)
    axs[0].plot(OAS.iloc[:, 0], color="darkblue", label="OAS")
    axs[0].plot(OAS.iloc[:, 1:], color="lightblue", alpha=0.1)
    axs[0].plot(
        OAS.iloc[:, 1:].mean(axis=1),
        color="darkblue",
        linestyle=":",
        label="Projected OAS",
        alpha=1.0,
    )
    axs[0].axhline(
        y=S_OAS_inf, color="darkblue", linestyle="--", label="Reversion Level"
    )
    axs[0].set_title("Monte Carlo Simulation of OAS")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("OAS")
    axs[0].legend()

    # Plot OAS trailing 3x simulation period
    OAS_trail = OAS.iloc[-len(paths_OAS) * 3 :]
    axs[1].plot(OAS_trail.iloc[:, 0], color="darkblue", label="OAS")
    axs[1].plot(OAS_trail.iloc[:, 1:], color="lightblue", alpha=0.1)
    axs[1].plot(
        OAS_trail.iloc[:, 1:].mean(axis=1),
        color="darkblue",
        linestyle=":",
        label="Projected OAS",
        alpha=1.0,
    )
    axs[1].axhline(
        y=S_OAS_inf, color="darkblue", linestyle="--", label="Reversion Level"
    )
    axs[1].set_title("Monte Carlo Simulation of OAS - Zoomed In")
    axs[1].set_xlabel("")
    axs[1].set_ylabel("OAS")
    axs[1].legend()

    # Plot Convexity
    paths_C = paths["cvx"]
    C = pd.concat([cvx_data, paths_C], axis=1)
    axs[2].plot(C.iloc[:, 0], color="darkgreen", label="Convexity")
    axs[2].plot(C.iloc[:, 1:], color="lightgreen", alpha=0.1)
    axs[2].plot(
        C.iloc[:, 1:].mean(axis=1),
        color="darkgreen",
        linestyle=":",
        label="Projected Convexity",
    )
    axs[2].axhline(y=C_inf, color="darkgreen", linestyle="--", label="Reversion Level")
    axs[2].set_title("Monte Carlo Simulation of Convexity")
    axs[2].set_xlabel("")
    axs[2].set_ylabel("Convexity")
    axs[2].legend()

    # Plot Convexity trailing 3x simulation period
    C_trail = C.iloc[-len(paths_C) * 3 :]
    axs[3].plot(C_trail.iloc[:, 0], color="darkgreen", label="Convexity")
    axs[3].plot(C_trail.iloc[:, 1:], color="lightgreen", alpha=0.1)
    axs[3].plot(
        C_trail.iloc[:, 1:].mean(axis=1),
        color="darkgreen",
        linestyle=":",
        label="Projected Convexity",
    )
    axs[3].axhline(y=C_inf, color="darkgreen", linestyle="--", label="Reversion Level")
    axs[3].set_title("Monte Carlo Simulation of Convexity - Zoomed In")
    axs[3].set_xlabel("")
    axs[3].set_ylabel("Convexity")
    axs[3].legend()

    fig.tight_layout()
    plt.show()

    return fig


def model_vs_actual(
    model: JointReversionModel,
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    simulation_freq: str,
    simulation_steps: int,
    num_paths: int,
    seed: int,
    step_interval: int = 1,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Series], list[pd.Series]]:
    """Compare model expected vs actual change in OAS and Convexity.

    Args:
        model (JointReversionModel): Model object.
        oas_data (pd.Series): Historical data for OAS.
        cvx_data (pd.Series): Historical data for Convexity.
        sigma_r_data (pd.Series): Historical data for Rates Volatility.
        nu_r_data (pd.Series): Historical data for Rates Volatility of Volatility.
        simulation_freq (str): Frequency for Monte Carlo simulation.
        simulation_steps (int): Number of simulation steps.
        num_paths (int): Number of Monte Carlo paths.
        seed (int): Seed for reproducibility.
        step_interval (int, optional): Interval for simulation steps. Defaults to 1.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[pd.Series], list[pd.Series]]: DataFrames for OAS and Convexity expected vs actual change.
    """
    # Perform Monte Carlo simulation
    # Time steps for estimation and simulation

    S_OAS_inf = float(np.mean(oas_data))
    C_inf = float(np.mean(cvx_data))

    oas_expected_paths = []
    cvx_expected_paths = []

    oas_expected = []
    cvx_expected = []

    oas_actual_start = []
    cvx_actual_start = []

    oas_actual_end = []
    cvx_actual_end = []

    oas_expected_change = []
    cvx_expected_change = []

    oas_actual_change = []
    cvx_actual_change = []

    # Get the last start date as a location in oas_data.index
    final_simulation_dates = pd.date_range(
        end=oas_data.index[-1], periods=simulation_steps, freq=simulation_freq
    )
    final_start = oas_data.index.get_loc(final_simulation_dates[0])

    for start in oas_data.index[:final_start:step_interval]:
        if verbose:
            print(start.strftime(DATE_FORMAT))

        simulation_dates = pd.date_range(
            start=start, periods=simulation_steps, freq=simulation_freq
        )
        end = simulation_dates[-1]

        paths = model.monte_carlo_simulation(
            S_OAS_init=oas_data.loc[start],
            C_init=cvx_data.loc[start],
            S_OAS_inf=S_OAS_inf,
            C_inf=C_inf,
            sigma_r_forward=sigma_r_data.loc[simulation_dates],
            nu_r_forward=nu_r_data.loc[simulation_dates],
            simulation_dates=simulation_dates,
            num_paths=num_paths,
            seed=seed,
            verbose=False,
        )
        oas_expected_path = paths["oas"].mean(axis=1)
        cvx_expected_path = paths["cvx"].mean(axis=1)

        oas_expected_paths.append(oas_expected_path)
        cvx_expected_paths.append(cvx_expected_path)

        oas_expected.append((end, float(oas_expected_path.iloc[-1])))
        cvx_expected.append((end, float(cvx_expected_path.iloc[-1])))

        oas_actual_start.append((start, float(oas_data.loc[start])))
        cvx_actual_start.append((start, float(cvx_data.loc[start])))

        oas_actual_end.append((end, float(oas_data.loc[end])))
        cvx_actual_end.append((end, float(cvx_data.loc[end])))

        oas_expected_change.append(
            (end, float(oas_expected_path.iloc[-1]) - float(oas_data.loc[start]))
        )
        cvx_expected_change.append(
            (end, float(cvx_expected_path.iloc[-1]) - float(cvx_data.loc[start]))
        )

        oas_actual_change.append(
            (end, float(oas_data.loc[end]) - float(oas_data.loc[start]))
        )
        cvx_actual_change.append(
            (end, float(cvx_data.loc[end]) - float(cvx_data.loc[start]))
        )

    oas_expected = pd.DataFrame(
        oas_expected, columns=["Date", "Expected End"]
    ).set_index("Date")
    cvx_expected = pd.DataFrame(
        cvx_expected, columns=["Date", "Expected End"]
    ).set_index("Date")

    oas_actual_start = pd.DataFrame(
        oas_actual_start, columns=["Date", "Actual Start"]
    ).set_index("Date")
    cvx_actual_start = pd.DataFrame(
        cvx_actual_start, columns=["Date", "Actual Start"]
    ).set_index("Date")

    oas_actual_end = pd.DataFrame(
        oas_actual_end, columns=["Date", "Actual End"]
    ).set_index("Date")
    cvx_actual_end = pd.DataFrame(
        cvx_actual_end, columns=["Date", "Actual End"]
    ).set_index("Date")

    oas_expected_change = pd.DataFrame(
        oas_expected_change, columns=["Date", "Expected Change"]
    ).set_index("Date")
    cvx_expected_change = pd.DataFrame(
        cvx_expected_change, columns=["Date", "Expected Change"]
    ).set_index("Date")

    oas_actual_change = pd.DataFrame(
        oas_actual_change, columns=["Date", "Actual Change"]
    ).set_index("Date")
    cvx_actual_change = pd.DataFrame(
        cvx_actual_change, columns=["Date", "Actual Change"]
    ).set_index("Date")

    oas = pd.concat(
        [
            oas_actual_start,
            oas_actual_end,
            oas_expected,
            oas_actual_change,
            oas_expected_change,
        ],
        axis=1,
    )
    cvx = pd.concat(
        [
            cvx_actual_start,
            cvx_actual_end,
            cvx_expected,
            cvx_actual_change,
            cvx_expected_change,
        ],
        axis=1,
    )

    return oas, cvx, oas_expected_paths, cvx_expected_paths


def plot_model_vs_actual(
    oas: pd.DataFrame,
    cvx: pd.DataFrame,
) -> plt.Figure:
    """Plot model expected vs actual change in OAS and Convexity.

    Args:
        oas (pd.DataFrame): OAS DataFrame with actual and expected change.
        cvx (pd.DataFrame): Convexity DataFrame with actual and expected change.

    Returns:
        plt.Figure: Figure object with the plot.
    """

    # Plot OAS and Convexity expected vs actual change in scatter plot with regression line
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    axs = axs.flatten()

    oas_plot = oas.loc[:, ["Actual Change", "Expected Change"]].dropna()
    axs[0].scatter(oas_plot["Actual Change"], oas_plot["Expected Change"])
    # Add regression line
    x = oas_plot["Actual Change"]
    y = oas_plot["Expected Change"]
    m, b = np.polyfit(x, y, 1)
    axs[0].plot(x, m * x + b, color="red")
    # Print regression line equation
    axs[0].text(
        0.05, 0.95, f"y = {m:.2f}x + {b:.2f}", transform=axs[0].transAxes, va="top"
    )
    # Add R^2 to plot
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    axs[0].text(0.05, 0.9, f"R^2 = {r2:.2f}", transform=axs[0].transAxes, va="top")
    axs[0].set_title("OAS Expected vs Actual Change")
    axs[0].set_xlabel("Actual Change")
    axs[0].set_ylabel("Expected Change")

    cvx_plot = cvx.loc[:, ["Actual Change", "Expected Change"]].dropna()
    axs[1].scatter(cvx_plot["Actual Change"], cvx_plot["Expected Change"])
    # Add regression line
    x = cvx_plot["Actual Change"]
    y = cvx_plot["Expected Change"]
    m, b = np.polyfit(x, y, 1)
    axs[1].plot(x, m * x + b, color="red")
    # Print regression line equation in the top left corner
    axs[1].text(
        0.05, 0.95, f"y = {m:.2f}x + {b:.2f}", transform=axs[1].transAxes, va="top"
    )
    # Add R^2 to plot
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    axs[1].text(0.05, 0.9, f"R^2 = {r2:.2f}", transform=axs[1].transAxes, va="top")
    axs[1].set_title("Convexity Expected vs Actual Change")
    axs[1].set_xlabel("Actual Change")
    axs[1].set_ylabel("Expected Change")

    plt.show()

    return fig


def main() -> None:
    """Main function running the MBS valuation process.

    The main function performs the following steps:
    1. Generate historical data.
    2. Generate forward data.
    3. Run MBS valuation model.
    4. Plot historical data and Monte Carlo paths.
    5. Compare model expected vs actual change in OAS and Convexity.
    6. Plot model vs actual.
    """
    # Set seed for reproducibility
    seed = 42

    # Define training data parameters
    zv_params = {"mu": 50, "theta": 0.01, "sigma": 20, "X0": 80}
    oas_params = {"mu": 30, "theta": 0.02, "sigma": 10, "X0": 20}
    zv_oas_rho = 0.8
    sigma_r_params = {"mu": 20, "theta": 0.01, "sigma": 10, "X0": 20}
    nu_r_params = {"mu": 10, "theta": 0.01, "sigma": 10, "X0": 10}

    # Mock data simulation parameters
    train_start_date = "2013-01-01"
    train_end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    historical_freq = "B"
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
    oas_data = oas_hist.copy()
    cvx_data = cvx_hist.copy()
    sigma_r_data = sigma_r_hist.copy()
    nu_r_data = nu_r_hist.copy()
    estimation_freq = "W-FRI"
    simulation_freq = "W-FRI"
    simulation_steps = 52
    num_paths = 100  # Number of Monte Carlo paths

    # Update X0 for forward data
    sigma_r_params["X0"] = sigma_r_data.iloc[-1]
    nu_r_params["X0"] = nu_r_data.iloc[-1]

    # Generate forward data
    start_date = oas_data.index[-1]
    simulation_dates = pd.date_range(
        start=start_date, periods=simulation_steps, freq=simulation_freq
    )
    sigma_r_forward, nu_r_forward = generate_forward_data(
        sigma_r_params,
        nu_r_params,
        simulation_dates,
        seed,
    )

    # Model parameters
    enable_spread_cvx = True
    enable_rate_vol = True
    enable_local_vol = True
    enable_mle = False
    model_param_overrides = {
        # "kappa": 0.6,
        # "lambda_": 0.6,
    }
    simulation_param_overrides = {
        # "sigma_r_forward": sigma_r_forward,
        # "nu_r_forward": nu_r_forward,
    }

    # Run MBS valuation model
    paths, model = run_value_model(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        enable_spread_cvx,
        enable_rate_vol,
        enable_local_vol,
        enable_mle,
        estimation_freq,
        simulation_freq,
        simulation_steps,
        num_paths,
        seed,
        model_param_overrides=model_param_overrides,
        simulation_param_overrides=simulation_param_overrides,
        verbose=True,
    )

    # Plot historical data and Monte Carlo paths
    S_OAS_inf = float(np.mean(oas_data))
    C_inf = float(np.mean(cvx_data))
    fig = plot_value_model(oas_data, cvx_data, paths, S_OAS_inf, C_inf)

    # Model vs Actual
    oas, cvx, oas_expected_paths, cvx_expected_paths = model_vs_actual(
        model,
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        simulation_freq,
        simulation_steps,
        num_paths,
        seed,
        step_interval=255,
        verbose=True,
    )

    # Plot model vs actual
    fig = plot_model_vs_actual(oas, cvx)


if __name__ == "__main__":
    main()
