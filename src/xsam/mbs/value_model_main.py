import numpy as np
import pandas as pd

from xsam import save
from xsam.mbs.mock_data import generate_forward_data, generate_historical_data
from xsam.mbs.value_model_eval import (
    plot_model_vs_actual,
    plot_value_model,
    run_model_vs_actual,
    run_value_model,
    summarize_ols,
)


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

    # Summarize OLS results
    ols_results = summarize_ols(model)

    # Plot historical data and Monte Carlo paths
    S_OAS_inf = float(np.mean(oas_data))
    C_inf = float(np.mean(cvx_data))
    fig = plot_value_model(oas_data, cvx_data, paths, S_OAS_inf, C_inf)

    save(fig, "value", "svg", add_timestamp=False)

    # Model vs Actual
    oas, cvx, *_, model = run_model_vs_actual(
        model,
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        simulation_freq,
        simulation_steps,
        num_paths,
        seed,
        step_interval=3 * 21,
        verbose=True,
    ).values()

    # Plot model vs actual
    fig = plot_model_vs_actual(oas, cvx, model.get_num_predictors())

    save(fig, "model_vs_actual", "svg", add_timestamp=False)


if __name__ == "__main__":
    main()
