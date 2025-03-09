import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller


class JointReversionModel:
    """Joint Reversion Model for OAS and Convexity.

    The model is defined by the following stochastic differential equations:
    dS_OAS = kappa * (S_OAS_inf - S_OAS) * dt + (gamma_0 * C + gamma_1 * sigma_r + gamma_2 * nu_r) * dt + sigma_O * dW_O
    dC = lambda * (C_inf - C) * dt + (beta_0 * S_OAS + beta_1 * sigma_r + beta_2 * nu_r) * dt + sigma_C * dW_C
    where:
    - S_OAS is the OAS spread.
    - C is the Convexity.
    - kappa and lambda are the reversion speeds for OAS and Convexity, respectively.
    - gamma and beta are the interest rate interaction coefficients for OAS and Convexity, respectively.
    - sigma_O is the volatility of OAS.
    - delta is the convexity volatility coefficient.
    - sigma_C is the volatility of Convexity.
    - dt is the time step.
    - dW_O and dW_C are Wiener processes for OAS and Convexity, respectively.

    The model is simulated using the Euler-Maruyama method with the following parameters:
    - S_OAS_init: Initial value of OAS spread.
    - C_init: Initial value of Convexity.
    - S_OAS_inf: Long-term mean of OAS spread.
    - C_inf: Convexity of the current coupon bond (or TBA).
    - sigma_r: Volatility of interest rates.
    - nu_r: Volatility of volatility of interest rates.
    - steps: Number of time steps.
    - seed: Random seed for reproducibility.

    For educational and illustrative purposes only. Not intended for trading or investment purposes.
    Use at your own risk. No warranty or guarantee of accuracy or reliability.
    """
    def __init__(
        self,
        dt: float,
        kappa: float = 0.0,
        gamma: list[float] = [0.0, 0.0, 0.0],
        sigma_O_0: float = 0.0,
        delta: float = 0.0,
        lambda_: float = 0.0,
        beta: list[float] = [0.0, 0.0, 0.0],
        sigma_C: float = 0.0,
    ):
        """Initialize the Joint Reversion Model.

        Args:
            dt (float): Simulation time step size. Usually 1/steps.
            kappa (float): Reversion speed for OAS.
            gamma (list[float]): Interest rate interaction coefficients for OAS.
            sigma_O_0 (float): Initial volatility of OAS.
            delta (float): Convexity volatility coefficient.
            lambda_ (float): Reversion speed for Convexity.
            beta (list[float]): Interest rate interaction coefficients for Convexity.
            sigma_C (float): Volatility of Convexity.
        """
        self.dt = dt
        self.kappa = kappa
        self.lambda_ = lambda_
        self.gamma = gamma
        self.beta = beta
        self.sigma_O_0 = sigma_O_0
        self.delta = delta
        self.sigma_C = sigma_C

    def get_parameters(self) -> dict[str, float | list[float]]:
        """Get the parameters of the Joint Reversion Model.

        Returns:
            dict[str, float | list[float]]: Model parameters.
        """
        return {
            "dt": self.dt,
            "kappa": self.kappa,
            "gamma": self.gamma,
            "sigma_O_0": self.sigma_O_0,
            "delta": self.delta,
            "lambda": self.lambda_,
            "beta": self.beta,
            "sigma_C": self.sigma_C,
        }
    
    def update_parameters(
        self,
        dt: float = None,
        kappa: float = None,
        gamma: list[float] = None,
        sigma_O_0: float = None,
        delta: float = None,
        lambda_: float = None,
        beta: list[float] = None,
        sigma_C: float = None,
    ) -> None:
        """Update the parameters of the Joint Reversion Model.

        Args:
            dt (float): Simulation time step size. Usually 1/steps.
            kappa (float): Reversion speed for OAS.
            gamma (list[float]): Interest rate interaction coefficients for OAS.
            sigma_O_0 (float): Initial volatility of OAS.
            delta (float): Convexity volatility coefficient.
            lambda_ (float): Reversion speed for Convexity.
            beta (list[float]): Interest rate interaction coefficients for Convexity.
            sigma_C (float): Volatility of Convexity.
        """
        if dt is not None:
            self.dt = dt
        if kappa is not None:
            self.kappa = kappa
        if gamma is not None:
            self.gamma = gamma
        if sigma_O_0 is not None:
            self.sigma_O_0 = sigma_O_0
        if delta is not None:
            self.delta = delta
        if lambda_ is not None:
            self.lambda_ = lambda_
        if beta is not None:
            self.beta = beta
        if sigma_C is not None:
            self.sigma_C = sigma_C

    def estimate_parameters_ols(
        self,
        S_OAS: pd.Series,
        C: pd.Series,
        sigma_r: pd.Series,
        nu_r: pd.Series,
        S_OAS_inf: float,
        C_inf: float,
        steps: int,
        enable_convexity: bool = True,
        enable_volatility: bool = True,
        verbose: bool = True,
    ) -> None:
        """Estimate the parameters of the joint reversion model using OLS.

        Args:
            S_OAS (pd.Series): OAS spread time series.
            C (pd.Series): Convexity time series.
            sigma_r (pd.Series): Volatility of interest rates time series.
            nu_r (pd.Series): Volatility of volatility of interest rates time series.
            S_OAS_inf (float): Long-term mean of OAS spread.
            C_inf (float): Convexity of the current coupon bond (or TBA).
            enable_convexity (bool, optional): Enable convexity interaction. Defaults to True.
            enable_volatility (bool, optional): Enable interest rate interaction. Defaults to True.
            verbose (bool, optional): Print estimated parameters. Defaults to True.
        """
        # Convert pd.Series to np.ndarray for better performance
        S_OAS = S_OAS.values
        C = C.values
        sigma_r = sigma_r.values
        nu_r = nu_r.values

        # Mean Reversion Parameter Estimation using OLS
        X_OAS = [S_OAS_inf - S_OAS[:-steps]]
        X_C = [C_inf - C[:-steps]]

        if enable_convexity:
            X_OAS.append(C[:-steps])
            X_C.append(S_OAS[:-steps])
        if enable_volatility:
            X_OAS.extend([sigma_r[:-steps], nu_r[:-steps]])
            X_C.extend([sigma_r[:-steps], nu_r[:-steps]])

        X_OAS = np.vstack(X_OAS).T
        y_OAS = S_OAS[steps:] - S_OAS[:-steps]
        X_C = np.vstack(X_C).T
        y_C = C[steps:] - C[:-steps]

        # OLS regression for OAS
        ols_OAS = np.linalg.lstsq(X_OAS, y_OAS, rcond=None)[0]
        kappa = ols_OAS[0]
        if not enable_convexity and not enable_volatility:
            gamma = [0.0, 0.0, 0.0]
        elif enable_convexity and not enable_volatility:
            gamma = [ols_OAS[1]] + [0.0, 0.0]
        elif not enable_convexity and enable_volatility:
            gamma = [0.0] + list(ols_OAS[1:])
        else:
            gamma = list(ols_OAS[1:])

        # OLS regression for Convexity
        ols_C = np.linalg.lstsq(X_C, y_C, rcond=None)[0]
        lambda_ = ols_C[0]
        if not enable_convexity and not enable_volatility:
            beta = [0.0, 0.0, 0.0]
        elif enable_convexity and not enable_volatility:
            beta = [ols_C[1]] + [0.0, 0.0]
        elif not enable_convexity and enable_volatility:
            beta = [0.0] + list(ols_C[1:])
        else:
            beta = list(ols_C[1:])

        # Residuals for variance calibration
        residuals_OAS = y_OAS - X_OAS @ ols_OAS
        residuals_C = y_C - X_C @ ols_C

        # Residual Variance Calibration
        def variance_function(params):
            sigma_O_0, delta = params
            return np.sum((residuals_OAS**2 - (sigma_O_0**2 + delta * C[:-steps] ** 2)) ** 2)

        if enable_convexity:
            initial_guess = self.sigma_O_0, self.delta
            sigma_O_0, delta = minimize(variance_function, initial_guess).x
        else:
            sigma_O_0 = np.std(residuals_OAS)
            delta = 0.0

        sigma_C = np.std(residuals_C)

        if verbose:
            print('\nEstimated Parameters (OLS):')
            print(f'kappa: {kappa:.4f}')
            print(f'gamma: {", ".join(f"{g:.4f}" for g in gamma)}')
            print(f'sigma_O_0: {sigma_O_0:.4f}')
            print(f'delta: {delta:.4f}')
            print(f'lambda: {lambda_:.4f}')
            print(f'beta: {", ".join(f"{b:.4f}" for b in beta)}')
            print(f'sigma_C: {sigma_C:.4f}')

        self.kappa = kappa
        self.lambda_ = lambda_
        self.gamma = gamma
        self.beta = beta
        self.sigma_O_0 = sigma_O_0
        self.delta = delta
        self.sigma_C = sigma_C

    def estimate_parameters_mle(
        self,
        S_OAS: pd.Series,
        C: pd.Series,
        sigma_r: pd.Series,
        nu_r: pd.Series,
        S_OAS_inf: float,
        C_inf: float,
        dt: float,
        enable_convexity: bool = True,
        enable_volatility: bool = True,
        verbose: bool = True,
    ) -> None:
        """Estimate the parameters of the joint reversion model using MLE.

        Args:
            S_OAS (pd.Series): OAS spread time series.
            C (pd.Series): Convexity time series.
            sigma_r (pd.Series): Volatility of interest rates time series.
            nu_r (pd.Series): Volatility of volatility of interest rates time series.
            S_OAS_inf (float): Long-term mean of OAS spread.
            C_inf (float): Convexity of the current coupon bond (or TBA).
            dt (float): Time step size.
            initial_guess (tuple[float, ..., float], optional): Initial guess for the parameters. Defaults to (0.1, ..., 0.1).
            enable_convexity (bool, optional): Enable convexity interaction. Defaults to True.
            enable_volatility (bool, optional): Enable interest rate interaction. Defaults to True.
            verbose (bool, optional): Print estimated parameters. Defaults to True.
        """
        S_OAS = S_OAS.values
        C = C.values
        sigma_r = sigma_r.values
        nu_r = nu_r.values

        def log_likelihood(params):
            kappa = params[0]
            gamma = params[1:4]
            sigma_O_0 = params[4]
            delta = params[5]
            lambda_ = params[6]
            beta = params[7:10]
            sigma_C = params[10]

            n = len(S_OAS)
            log_likelihood = 0

            for t in range(1, n):
                S_OAS_prev = S_OAS[t - 1]
                C_prev = C[t - 1]
                sigma_r_prev = sigma_r[t - 1]
                nu_r_prev = nu_r[t - 1]

                S_OAS_mean = S_OAS_prev + kappa * (S_OAS_inf - S_OAS_prev) * dt
                if enable_convexity:
                    S_OAS_mean += gamma[0] * C_prev * dt
                if enable_volatility:
                    S_OAS_mean += (gamma[1] * sigma_r_prev + gamma[2] * nu_r_prev) * dt

                sigma_O = (
                    np.sqrt(sigma_O_0**2 + delta * C_prev**2)
                    if enable_convexity
                    else sigma_O_0
                )

                C_mean = C_prev + lambda_ * (C_inf - C_prev) * dt
                if enable_convexity:
                    C_mean += beta[0] * S_OAS_prev * dt
                if enable_volatility:
                    C_mean += (beta[1] * sigma_r_prev + beta[2] * nu_r_prev) * dt

                log_likelihood += -0.5 * np.log(2 * np.pi * sigma_O**2 * dt) - 0.5 * (
                    (S_OAS[t] - S_OAS_mean) ** 2 / (sigma_O**2 * dt)
                )
                log_likelihood += -0.5 * np.log(2 * np.pi * sigma_C**2 * dt) - 0.5 * (
                    (C[t] - C_mean) ** 2 / (sigma_C**2 * dt)
                )

            return -log_likelihood

        # Optimize the log-likelihood function
        initial_guess = (
            self.kappa,
            *self.gamma,
            self.sigma_O_0,
            self.delta,
            self.lambda_,
            *self.beta,
            self.sigma_C,
        )
        result = minimize(log_likelihood, initial_guess, method='L-BFGS-B')

        # Extract the estimated parameters
        kappa = result.x[0]
        gamma = result.x[1:4]
        sigma_O_0 = result.x[4]
        delta = result.x[5]
        lambda_ = result.x[6]
        beta = result.x[7:10]
        sigma_C = result.x[10]

        if verbose:
            print('\nEstimated Parameters (MLE):')
            print(f'kappa: {kappa:.4f}')
            print(f'gamma: {", ".join(f"{g:.4f}" for g in gamma)}')
            print(f'sigma_O_0: {sigma_O_0:.4f}')
            print(f'delta: {delta:.4f}')
            print(f'lambda: {lambda_:.4f}')
            print(f'beta: {", ".join(f"{b:.4f}" for b in beta)}')
            print(f'sigma_C: {sigma_C:.4f}')

        self.kappa = kappa
        self.gamma = gamma
        self.sigma_O_0 = sigma_O_0
        self.delta = delta
        self.lambda_ = lambda_
        self.beta = beta
        self.sigma_C = sigma_C
    
    @staticmethod
    def stationarity_tests(
        S_OAS: pd.Series,
        C: pd.Series,
        verbose: bool = True,
    ) -> tuple[
        tuple[float, float, int, int, dict, float],
        tuple[float, float, int, int, dict, float],
    ]:
        """Test the stationarity of the time series using the Augmented Dickey-Fuller test.

        Args:
            S_OAS (pd.Series): OAS spread time series.
            C (pd.Series): Convexity time series.

        Returns:
            tuple[ tuple[float, float, int, int, dict, float], tuple[float, float, int, int, dict, float], ]: ADF test results for OAS and Convexity.

        ADF test results:
        - Test statistic.
        - P-value.
        - Number of lags used.
        - Number of observations used.
        - Critical values.
        - Maximum information criterion.

        ADF test null hypothesis:
        - H0: The time series is non-stationary.

        ADF test alternative hypothesis:
        - H1: The time series is stationary.

        If the p-value is less than the significance level (e.g., 0.05), we reject the null hypothesis and conclude that the time series is stationary.

        Reference:
        - https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
        """
        adf_OAS = adfuller(S_OAS)
        adf_C = adfuller(C)

        if verbose:
            # Print ADF test results
            print('\nAugmented Dickey-Fuller Test for OAS:')
            print(f'Test Statistic: {adf_OAS[0]:.4f}')
            print(f'P-Value: {adf_OAS[1]:.4f}')
            print(f'Lags Used: {adf_OAS[2]:.0f}')
            print(f'Observations Used: {adf_OAS[3]:.0f}')
            print('Critical Values:')
            for key, value in adf_OAS[4].items():
                print(f'  {key}: {value:.4f}')
            print(f'Max Information Criteria: {adf_OAS[5]:.4f}')

            print('\nAugmented Dickey-Fuller Test for Convexity:')
            print(f'Test Statistic: {adf_C[0]:.4f}')
            print(f'P-Value: {adf_C[1]:.4f}')
            print(f'Lags Used: {adf_C[2]:.0f}')
            print(f'Observations Used: {adf_C[3]:.0f}')
            print('Critical Values:')
            for key, value in adf_C[4].items():
                print(f'  {key}: {value:.4f}')
            print(f'Max Information Criteria: {adf_C[5]:.4f}')

        return adf_OAS, adf_C

    def simulate(
        self,
        S_OAS_init: float,
        C_init: float,
        S_OAS_inf: float,
        C_inf: float,
        sigma_r: pd.Series,
        nu_r: pd.Series,
        simulation_dates: pd.DatetimeIndex,
        enable_convexity: bool = True,
        enable_volatility: bool = True,
        rng: np.random.Generator = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Simulate the joint reversion model for OAS and Convexity.

        Args:
            S_OAS_init (float): Initial value of OAS spread.
            C_init (float): Initial value of Convexity.
            S_OAS_inf (float): Long-term mean of OAS spread.
            C_inf (float): Convexity of the current coupon bond (or TBA).
            sigma_r (pd.Series): Volatility of interest rates.
            nu_r (pd.Series): Volatility of volatility of interest rates.
            simulation_dates (pd.DatetimeIndex): Dates for the simulation.
            enable_convexity (bool, optional): Enable convexity interaction. Defaults to True.
            enable_volatility (bool, optional): Enable interest rate interaction. Defaults to True.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.

        Returns:
            tuple[pd.Series, pd.Series, pd.Series]: Simulated OAS, Convexity, and Volatility of OAS.
        """
        if rng is None:
            rng = np.random.default_rng()

        steps = len(simulation_dates)
        S_OAS = pd.Series(index=simulation_dates)
        C = pd.Series(index=simulation_dates)
        sigma_O = pd.Series(index=simulation_dates)
        S_OAS.iloc[0] = S_OAS_init
        C.iloc[0] = C_init
        sigma_O.iloc[0] = np.sqrt(self.sigma_O_0**2 + self.delta * C_init**2)

        for t in range(1, steps):
            Z_O = rng.normal()
            Z_C = rng.normal()

            if enable_convexity:
                gamma_term = self.gamma[0] * float(C.iloc[t - 1])
                beta_term = self.beta[0] * float(S_OAS.iloc[t - 1])
            else:
                gamma_term = 0
                beta_term = 0

            if enable_volatility:
                gamma_term += (
                    self.gamma[1] * float(sigma_r.iloc[t - 1]) 
                    + self.gamma[2] * float(nu_r.iloc[t - 1])
                )
                beta_term += (
                    self.beta[1] * float(sigma_r.iloc[t - 1]) 
                    + self.beta[2] * float(nu_r.iloc[t - 1])
                )

            S_OAS.iloc[t] = (
                float(S_OAS.iloc[t - 1])
                + self.kappa * (S_OAS_inf - float(S_OAS.iloc[t - 1])) * self.dt
                + gamma_term * self.dt
                + sigma_O.iloc[t - 1] * np.sqrt(self.dt) * Z_O
            )

            C.iloc[t] = (
                float(C.iloc[t - 1])
                + self.lambda_ * (C_inf - float(C.iloc[t - 1])) * self.dt
                + beta_term * self.dt
                + self.sigma_C * np.sqrt(self.dt) * Z_C
            )

            sigma_O.iloc[t] = np.sqrt(self.sigma_O_0**2 + self.delta * C.iloc[t]**2)

        return S_OAS, C, sigma_O


    def monte_carlo_simulation(
        self,
        S_OAS_init: float,
        C_init: float,
        S_OAS_inf: float,
        C_inf: float,
        sigma_r_forward: float | pd.Series,
        nu_r_forward: float | pd.Series,
        simulation_dates: pd.DatetimeIndex,
        enable_convexity: bool = True,
        enable_volatility: bool = True,
        num_paths: int = 1000,
        seed: int = None,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Perform Monte Carlo simulation of the Joint Reversion Model.

        Args:
            model (JointReversionModel): Joint Reversion Model instance.
            S_OAS_init (float): Initial value of OAS spread.
            C_init (float): Initial value of Convexity.
            C_inf (float): Convexity of the current coupon bond (or TBA).
            sigma_r_forward (float | pd.Series): Volatility of interest rates.
            nu_r_forward (float | pd.Series): Volatility of volatility of interest rates.
            simulation_dates (pd.DatetimeIndex): Dates for the simulation.
            enable_convexity (bool, optional): Enable convexity interaction. Defaults to True.
            enable_volatility (bool, optional): Enable interest rate interaction. Defaults to True.
            num_paths (int): Number of Monte Carlo paths.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            dict[str, pd.DataFrame]: Simulated paths of OAS, Convexity, and Volatility of OAS.

        The Monte Carlo simulation generates multiple paths of the OAS spread, Convexity, and Volatility of OAS using the Joint Reversion Model.
        The expected value of OAS, Convexity, and Volatility of OAS can be calculated as the average of the simulated paths.

        Reference:
        - https://en.wikipedia.org/wiki/Monte_Carlo_method

        Note:
        - The Monte Carlo simulation is a stochastic process that generates random paths.
        - The results may vary each time the simulation is run.
        - The results are based on the model assumptions and parameters.
        - The results are for illustrative and educational purposes only.
        - Use at your own risk. No warranty or guarantee of accuracy or reliability.
        """
        paths_OAS = []
        paths_C = []
        paths_sigma_O = []

        steps = len(simulation_dates)

        if isinstance(sigma_r_forward, float):
            sigma_r_series = pd.Series(np.full(steps, sigma_r_forward), index=simulation_dates)
        else:
            sigma_r_series = sigma_r_forward

        if isinstance(nu_r_forward, float):
            nu_r_series = pd.Series(np.full(steps, nu_r_forward), index=simulation_dates)
        else:
            nu_r_series = nu_r_forward

        rng = np.random.default_rng(seed)

        for i in range(num_paths):
            S_OAS, C, sigma_O = self.simulate(
                S_OAS_init,
                C_init,
                S_OAS_inf,
                C_inf,
                sigma_r_series,
                nu_r_series,
                simulation_dates,
                enable_convexity,
                enable_volatility,
                rng,
            )
            paths_OAS.append(S_OAS)
            paths_C.append(C)
            paths_sigma_O.append(sigma_O)

        if verbose:
            final_oas = np.mean([path.iloc[-1] for path in paths_OAS])
            final_cvx = np.mean([path.iloc[-1] for path in paths_C])
            final_sigma_o = np.mean([path.iloc[-1] for path in paths_sigma_O])
            print(f"\nMonte Carlo Simulation ({num_paths} paths):")
            print(f"Final expected value of OAS: {final_oas:.0f} bps")
            print(f"Final expected value of Convexity: {final_cvx:.0f} bps")
            print(f"Final expected value of Sigma_O: {final_sigma_o:.0f} bps")

        return {
            "oas": pd.DataFrame(paths_OAS).T,
            "cvx": pd.DataFrame(paths_C).T,
            "sigma_o": pd.DataFrame(paths_sigma_O).T,
        }
