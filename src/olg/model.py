import math
import numpy as np
import pandas as pd
from scipy import optimize
from olg.utils import calc_expected_wages, calc_implied_capital
from olg.utils import make_a, calc_transfer_old


class Model:
    """Defines the model object for simulations of the economy under Cobb-Douglas technology

    Attributes
    ----------
    parameters : namedtuple
        The parameters that do not change with each calibration target
    rho : float
        Persistence of technology process
    with_endowment : boolean
        Equal to "True" when risk-free endowment is avaialble and "False" otherwise.
    growth_rate : float
        The annual growth rate of effective labor
    risky_rate : float
        The marginal product of capital, R
    riskfree_rate : float
        The risk free rate of interest, r
    values : 1-D ndarray
        The set possible levels of the standard normal variable from Gauss-Hermite quadrature
    weights : 1-D ndarray
        The weights assoicated with the possible levels in values
    a_values : dictionary of 1-D ndarrays
        The stochastic values of a for us in simulation, corresponding to each possible value of rho

    Methods
    -------
    calibrate(self)
        Sets the parameters gamma and beta
    _calc_gamma(self)
        Calculates gamma
    _calc_beta(self)
        Calculates beta
    _calc_beta_guess(self)
        Calculate initial guess at beta for scipy.optimize.fsolve()
    _risky_rate_deviation(self, beta, *args)
        Returns the difference bewteen a simulated risky rate level and the calibration target
    calc_transfer_to_wage_ratio(self, transfer_to_capital)
        Converts the transfer as a share of capital into a share of expected wages
    simulate(self, transfer=0, transfer_type='fixed')
        Simulates the economy
    _find_savings(self, period, transfer, transfer_type)
        Returns the level of savings consistent with the Euler equation and budget constraint.
    _savings_deviation(savings, *args)
        Returns the difference bewteen a candidate level of savings and the levels of savings
        it implies through the Euler equation and the budget constraint. It will equal 0 when the
        fsolve function finds a solution.
    store_results(self, transfer=0, transfer_type='none')
        Stores simulation results in a pandas dataframe.
    _calc_utilities(self, transfer=0, transfer_type='fixed')
        Calculate utilities in each generation for all periods
    _calc_avg_total_utility(self, chi)
        Calculates average total utility across generations.
    summarize_results(self)
        Creates pandas DataFrame to summarize results.
    """

    def __init__(self, parameters):
        self.parameters = parameters

        self.values, self.weights = np.polynomial.hermite.hermgauss(self.parameters.hermgauss_degree)
        self.values = np.reshape(self.values * math.sqrt(2), (1, self.parameters.hermgauss_degree))
        self.weights = np.reshape(self.weights / np.sum(self.weights), (self.parameters.hermgauss_degree, 1))

        np.random.seed(self.parameters.seed)
        epsilon = np.random.normal(0, 1, (parameters.n, 1))
        self.a_values = {}
        for rho in parameters.rho_values:
            self.a_values[rho] = make_a(self.parameters.mu, self.parameters.sigma, epsilon, rho)

        self.simulation = {
            'expected_log_a': np.zeros((self.parameters.n, 1)),
            'capital': np.zeros((self.parameters.n, 1)),
            'wage': np.zeros((self.parameters.n, 1)),
            'expected_wages': np.zeros((self.parameters.n, 1)),
            'endowment': np.zeros((self.parameters.n, 1)),
            'income': np.zeros((self.parameters.n, 1)),
            'risky_rate': np.zeros((self.parameters.n, 1)),
            'consumption_young': np.zeros((self.parameters.n, 1)),
            'transfer_young': np.zeros((self.parameters.n, 1)),
            'utility_young': np.zeros((self.parameters.n, 1)),
            'utility_old': np.zeros((self.parameters.n, 1)),
            'utility_total': np.zeros((self.parameters.n, 1))
        }

        self.simulation_results = pd.DataFrame()


    def calibrate(self):
        """Calculate gamma and beta parameters that are consistent with calibration target parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.gamma = self._calc_gamma(
            self.risky_rate,
            self.riskfree_rate,
            self.parameters.sigma,
            self.parameters.generation_length
        )

        self.beta = self._calc_beta()

        return None


    def _calc_gamma(self, risky_rate, riskfree_rate, sigma, generation_length):
        """Calculates gamma, the parameter measuring risk aversion.

        Parameters
        ----------
        risky_rate : float
            The marginal product of capital, R
        riskfree_rate : float
            The risk free rate of interest, r
        sigma : float
            Standard deviation of the technology process
        generation_length : int
            Length of generation, in years

        Returns
        -------
        float
        """

        return math.log(risky_rate / riskfree_rate) / (sigma ** 2) * generation_length


    def _calc_beta(self):
        """Calibrates beta given R, capital share, and other parameters

        Parameters
        ----------
        None

        Returns
        -------
        beta : float
            The weight placed on second generation utility.
        """

        beta_guess = self._calc_beta_guess()

        # If the endowment is available, income is double wages and the savings has to be half
        # what it would be otherwise to make the return on capital match the target risky rate.
        if self.with_endowment:
            beta_guess *= 0.5

        args = (self, self.risky_rate)

        calibrated_beta, infodict, ier, mesg = optimize.fsolve(
            self._risky_rate_deviation,
            beta_guess,
            args,
            full_output=True
        )

        if ier != 1:
            raise Exception('Problem with calibrated_beta in Model._calc_beta()')

        return calibrated_beta[0]


    def _calc_beta_guess(self):
        """"Calculate initial guess of beta for fsolve

        Parameters
        ----------
        None

        Returns
        ----------
        beta_guess: float
            An initial value for the weight placed on second generation utility
        """
        compounded_risky_rate = self.risky_rate ** self.parameters.generation_length

        return (self.parameters.b / (1 - self.parameters.b)) * (1 / compounded_risky_rate)


    def _risky_rate_deviation(self, beta, *args):
        """Returns the difference bewteen a simulated risky rate level and the calibration target.

        Parameters
        ----------
        beta : float
            The value of beta to be used in the simulation.
        *args : tuple
            Arguments to be used in the simulation.

        Returns
        -------
        deviation : float
            Deviation between simulated risky rate level and the calibration target.
        """
        (self, risky_rate) = args
        self.beta = beta
        self.simulate()

        estimated_risky_rate = np.mean(self.simulation['risky_rate'][self.parameters.transition_length:]).item()
        deviation = estimated_risky_rate - (risky_rate ** self.parameters.generation_length)

        return deviation


    def calc_transfer_to_wage_ratio(self, transfer_to_capital_ratio):
        """Converts the transfer as a share of capital into a share of expected wages.

        Parameters
        ----------
        transfer_to_capital_ratio : float
            The size of the transfer as a share of capital.

        Returns
        -------
        transfer_to_wage_ratio:
            The size of the transfer as a share of expected wages
        """

        avg_capital_to_wage_ratio = np.mean(
            self.simulation['capital'][self.parameters.transition_length:] /
            self.simulation['expected_wages'][self.parameters.transition_length:]
        )
        transfer_to_wage_ratio = transfer_to_capital_ratio * avg_capital_to_wage_ratio

        return transfer_to_wage_ratio


    def simulate(self, transfer=0, transfer_type='fixed'):
        a = self.a_values[self.rho]
        self.simulation['capital'][0] = calc_implied_capital(
            self.parameters.b,
            a[0],
            self.risky_rate ** self.parameters.generation_length
        )

        for period in range(self.parameters.n):
            self.simulation['expected_log_a'][period] = self.parameters.mu * (1 - self.rho) + math.log(a[period]) * self.rho
            self.simulation['wage'][period] = (1 - self.parameters.b) * a[period] * self.simulation['capital'][period] ** self.parameters.b
            self.simulation['risky_rate'][period] = self.parameters.b * a[period] * self.simulation['capital'][period] ** (self.parameters.b - 1)
            self.simulation['expected_wages'][period] = calc_expected_wages(
                self.parameters.b,
                math.exp(self.simulation['expected_log_a'][period] + self.parameters.sigma ** 2 / 2),
                self.risky_rate ** self.parameters.generation_length
            )

            if self.with_endowment:
                self.simulation['endowment'][period] = self.simulation['expected_wages'][period]
            else:
                self.simulation['endowment'][period] = 0

            if period > 0:
                self.simulation['transfer_young'][period] = transfer * self.simulation['expected_wages'][period - 1]
            else:
                self.simulation['transfer_young'][period] = transfer * calc_expected_wages(
                    self.parameters.b,
                    math.exp(self.parameters.mu + self.parameters.sigma ** 2 / 2),
                    self.risky_rate ** self.parameters.generation_length
                )

            if transfer_type == 'variable':
                self.simulation['transfer_young'][period] = (
                    self.simulation['transfer_young'][period] *
                    a[period] /
                    math.exp(self.simulation['expected_log_a'][period]
                    + self.parameters.sigma ** 2 / 2)
                )

            self.simulation['income'][period] = (
                self.simulation['wage'][period] +
                self.simulation['endowment'][period] -
                self.simulation['transfer_young'][period]
            )

            savings = self._find_savings(period, transfer, transfer_type)

            self.simulation['consumption_young'][period] = self.simulation['income'][period] - savings

            # Set capital in next period to savings in this one, divided by one plus growth rate
            if period < (self.parameters.n - 1):
                self.simulation['capital'][period + 1] = savings / ((1 + self.growth_rate) ** self.parameters.generation_length)


    def _find_savings(self, period, transfer, transfer_type):
        """Returns the level of savings consistent with the Euler equation and budget constraint.

        Parameters
        ----------
        period : int (unpacked from args)
            The period of the simulation in which savings is being calculated.
        transfer : float or 1-D array of floats
            The amount that the representative household will receive in old age from a transfer.
        next_a_values : 1-D array of floats
            The set of possible values of a in the next period.

        Returns
        -------
        savings_solution : float
            The level of savings that is consistenet with the budget constraint and Euler equation.
        """

        if transfer == 0:
            return self.beta * self.simulation['income'][period]

        else:
            savings_guess = self.beta * self.simulation['income'][period] \
                - (1 - self.beta) * transfer * self.simulation['expected_wages'][period]
            next_a_values = np.exp(self.values * self.parameters.sigma + self.simulation['expected_log_a'][period])
            transfer_values = transfer * self.simulation['expected_wages'][period]

            if transfer_type == 'variable':
                transfer_values = transfer_values * next_a_values / \
                    math.exp(self.simulation['expected_log_a'][period] + self.parameters.sigma ** 2 / 2)

            args = (self, period, next_a_values, transfer_values)

            savings_solution, infodict, ier, mesg = optimize.fsolve(
                self._savings_deviation,
                savings_guess,
                args,
                full_output=True
            )

            if ier != 1:
                raise Exception('Problem with savings_solution in Model._find_savings()')

        return savings_solution


    def _savings_deviation(self, savings, *args):
        """Returns the difference bewteen a candidate level of savings and the levels of savings
        it implies through the Euler equation and the budget constraint. It will equal 0 when the
        fsolve function finds a solution.

        Parameters
        ----------
        period : int (unpacked from args)
            The period of the simulation in which savings is being calculated.
        next_a_values : 1-D array of floats
            The set of possible values of a in the next period
        transfer : float or 1-D array of floats
            The amount that the representative household will receive in old age from a transfer.

        Returns
        ----------
        deviation : float
            Deviation between candidate value of savings and implied savings.
        """

        (self, period, next_a_values, transfer) = args
        growth_factor = (1 + self.growth_rate) ** self.parameters.generation_length
        capital = savings / growth_factor

        marginal_product_capital = next_a_values * (self.parameters.b * capital ** (self.parameters.b - 1))
        consumption_old = growth_factor * (next_a_values * (self.parameters.b  * savings ** self.parameters.b) + transfer)

        right_hand_side_top = np.matmul(marginal_product_capital * consumption_old ** - self.gamma, self.weights)
        right_hand_side_bottom = np.matmul(consumption_old** (1 - self.gamma), self.weights)

        # Euler equations implies consumption in time t = (1 - beta) / beta * right_hand_side_bottom / right_hand_side_top
        # so income - (1 - beta) / beta * right_hand_side_bottom / right_hand_side_top is implied savings
        implied_consumption_young = (1 - self.beta) / self.beta * right_hand_side_bottom / right_hand_side_top
        implied_savings = self.simulation['income'][period]  - implied_consumption_young
        deviation = implied_savings - savings

        return deviation.flatten()


    def store_results(self, transfer=0, transfer_type='none'):
        """Stores simulation results in a pandas DataFrame.

        Parameters
        ----------
        transfer : float
            The size of the transfer as a share of capital
        transfer_type : string
            The type of transfer: "none", "fixed" or "variable"

        Returns
        -------
        None
        """

        self._calc_utilities(transfer, transfer_type)

        for chi in self.parameters.chi_values:
            i = len(self.simulation_results.index)

            self.simulation_results.loc[i, 'transfer'] = float(transfer)
            self.simulation_results.loc[i, 'transfer_type'] = transfer_type
            self.simulation_results.loc[i, 'risky_rate'] = self.risky_rate
            self.simulation_results.loc[i, 'riskfree_rate'] = self.riskfree_rate
            self.simulation_results.loc[i, 'growth_rate'] = self.growth_rate
            self.simulation_results.loc[i, 'with_endowment'] = self.with_endowment
            self.simulation_results.loc[i, 'rho'] = self.rho
            self.simulation_results.loc[i, 'chi'] = float(chi)
            self.simulation_results.loc[i, 'gamma'] = self.gamma
            self.simulation_results.loc[i, 'beta'] = self.beta
            self.simulation_results.loc[i, 'average_capital'] = np.mean(self.simulation['capital'])
            self.simulation_results.loc[i, 'average_wage'] = np.mean(self.simulation['wage'])
            self.simulation_results.loc[i, 'average_consumption_young'] = np.mean(self.simulation['consumption_young'])
            self.simulation_results.loc[i, 'average_risky_rate'] = np.mean(self.simulation['risky_rate'])
            self.simulation_results.loc[i, 'average_income'] = np.mean(self.simulation['income'])
            self.simulation_results.loc[i, 'average_total_utility'] = self._calc_avg_total_utility(
                self.simulation['utility_total'],
                self.parameters.transition_length,
                chi
            )


    def _calc_utilities(self, transfer=0, transfer_type='fixed'):
        """Calculate utilities in each generation for all periods.

        Parameters
        ----------
        transfer : float
            The size of the transfer as a share of capital.
        transfer_type : string
            The type of transfer: "none", "fixed" or "variable".

        Returns
        -------
        None
        """

        # Create nsim x n matrix with a row of n repeated values for
        # median returns at old age in each simulated period
        capital = self.simulation['income'] - self.simulation['consumption_young']
        median_returns = np.matmul(np.exp(self.simulation['expected_log_a']) * self.parameters.b * capital ** self.parameters.b, np.ones((1, self.parameters.hermgauss_degree)))
        # Multiply median_returns by exp(values * sigma) to get a nsim x n matrix of returns
        returns_values = median_returns * np.exp(np.matmul(np.ones((self.parameters.n, 1)), self.values * self.parameters.sigma))

        # if there are no transfers, then consumption_old = returns
        transfer_values = calc_transfer_old(transfer, self.simulation['expected_wages'], transfer_type, self.values * self.parameters.sigma, self.parameters.sigma)
        consumption_old_values = (returns_values + transfer_values) * (1 + self.growth_rate) ** self.parameters.generation_length
        utility_old_values = consumption_old_values ** (1 - self.gamma)

        self.simulation['utility_old'] = self.beta / (1 - self.gamma) * np.log(np.matmul(utility_old_values, self.weights))
        self.simulation['utility_young'] = (1 - self.beta) * np.log(self.simulation['consumption_young'])
        self.simulation['utility_total'] = self.simulation['utility_old'] + self.simulation['utility_young']


    def _calc_avg_total_utility(self, total_utility, transition_length, chi):
        """Calculate average total utility over all simulations after transition period.

        Parameters
        ----------
        total_utility : 1-D np.array
            Vector of utilities of length n (number of generations in a simulation)
        transition_length : int
            Assumed number of generations before reaching steady state.
        chi : float
            Aversion to the risk that a generation will be born into a lifetime of low consumption.

        Returns
        -------
        float
            Average total utility
        """

        if chi == 1:
            return np.mean(np.exp(total_utility[transition_length:]))
        else:
            return np.mean(np.exp(total_utility[transition_length:]) ** (1 - chi)) ** (1 / (1 - chi))


    def calc_welfare_effects(self, sim_results):
        """Merge filtered simulation results DataFrames and calculate components of welfare effects:
        crowd_out_effect, risk_shifting_effect, and total_welfare_effect.

        Parameters
        ----------
        sim_results : pd.DataFrame
            DataFrame containing summarized simulation results.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with simulation summary and columns containing welfare effects.
        """

        without_transfer = sim_results[sim_results['transfer_type'] == 'none'].drop(columns=['transfer', 'transfer_type'])
        fixed_transfer = sim_results[sim_results['transfer_type'] == 'fixed'].drop(columns=['transfer_type', 'gamma', 'beta'])
        variable_transfer = sim_results[sim_results['transfer_type'] == 'variable'].drop(columns=['transfer', 'transfer_type', 'gamma', 'beta'])

        merge_cols = ['with_endowment', 'growth_rate', 'risky_rate', 'riskfree_rate', 'rho', 'chi']
        transfers = fixed_transfer.merge(variable_transfer, on=(merge_cols), suffixes=("_fixed", "_variable"))
        summary = without_transfer.merge(transfers, on=(merge_cols), suffixes=(None, None))

        summary['crowd_out_effect'] = (summary['average_total_utility_variable'] - summary['average_total_utility']) / summary['average_total_utility'] * 100
        summary['risk_shifting_effect'] = (summary['average_total_utility_fixed'] - summary['average_total_utility_variable']) / summary['average_total_utility'] * 100
        summary['total_welfare_effect'] = summary['crowd_out_effect'] + summary['risk_shifting_effect']

        return summary


class ModelLinear:
    """Defines the model object for simulations of the economy under linear technology

    Attributes
    ----------
    parameters : namedtuple
        The parameters that do not change with each calibration target
    values : 1-D np.ndarray
        The set possible levels of the standard normal variable from Gauss-Hermite quadrature
    weights : 1-D np.ndarray
        The weights assoicated with the possible levels in values
    mu : float
        The mean of the technology process
    gamma : float
        The parameter measuring the degree of risk aversion
    with_endowment : boolean
        Equal to "True" when risk-free endowment is avaialble and "False" otherwise.
    growth_rate : float
        The annual growth rate of effective labor
    risky_rate : float
        The marginal product of capital, R
    riskfree_rate : float
        The risk free rate of interest, r
    epsilon : 1-D np.ndarray
        A vector of random normal variables used in projecting the stochastic path of technological productivity

    Methods
    -------
    calibrate(self)
        Sets the parameters gamma and mu.
    _calc_gamma(self, risky_rate, riskfree_rate, sigma, generation_length)
        Calculates gamma, the parameter measuring risk aversion.
    _calc_mu(self, b, generation_length, risky_rate, sigma)
        Calculates mu, the average annual growth rate in the technology process.
    simulate(self, transfer=0, transfer_type='fixed')
        Simulates the economy
    _find_savings(self, period, transfer, transfer_type)
        Returns the level of savings consistent with the Euler equation and budget constraint.
    _savings_deviation(savings, *args)
        Returns the difference bewteen a candidate level of savings and the levels of savings
        it implies through the Euler equation and the budget constraint. It will equal 0 when the
        fsolve function finds a solution.
    _calc_utilities(self, transfer=0, transfer_type='fixed')
        Calculate utilities in each generation for all periods.
    calc_transfer_to_wage_ratio(self, transfer_to_capital_ratio)
        Converts the transfer as a share of capital into a share of expected wages
    store_results(self, transfer=0, transfer_type='none')
        Stores simulation results in a pandas DataFrame.
    _calc_avg_total_utility(self, total_utility, transition_length, chi)
        Calculates average total utility across generations
    calc_welfare_effects(self)
        Merge filtered simulation results DataFrames and calculate components of welfare effects
    """


    def __init__(self, parameters):
        self.parameters = parameters

        self.values, self.weights = np.polynomial.hermite.hermgauss(self.parameters.hermgauss_degree)
        self.values = np.reshape(self.values * math.sqrt(2), (1, self.parameters.hermgauss_degree))
        self.weights = np.reshape(self.weights / np.sum(self.weights), (self.parameters.hermgauss_degree, 1))

        np.random.seed(self.parameters.seed)
        self.epsilon = np.random.normal(0, 1, (parameters.n, 1))

        self.simulation = {
            'expected_log_a': np.zeros((self.parameters.n, 1)),
            'capital': np.zeros((self.parameters.n, 1)),
            'wage': np.zeros((self.parameters.n, 1)),
            'expected_wages': np.zeros((self.parameters.n, 1)),
            'endowment': np.zeros((self.parameters.n, 1)),
            'income': np.zeros((self.parameters.n, 1)),
            'risky_rate': np.zeros((self.parameters.n, 1)),
            'consumption_young': np.zeros((self.parameters.n, 1)),
            'transfer_young':np.zeros((self.parameters.n, 1)),
            'utility_young':np.zeros((self.parameters.n, 1)),
            'utility_old':np.zeros((self.parameters.n, 1)),
            'utility_total':np.zeros((self.parameters.n, 1))
        }

        self.simulation_results = pd.DataFrame()


    def calibrate(self):
        """Calculate parameters gamma and mu that are consistent with calibration target parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.gamma = self._calc_gamma(
             self.risky_rate,
             self.riskfree_rate,
             self.parameters.sigma,
             self.parameters.generation_length
        )

        self.mu = self._calc_mu(
            self.parameters.b,
            self.parameters.generation_length,
            self.risky_rate,
            self.parameters.sigma
        )

        self.a = make_a(
            self.mu,
            self.parameters.sigma,
            self.epsilon
        )

        return None


    def _calc_gamma(self, risky_rate, riskfree_rate, sigma, generation_length):
        """Calculates gamma, the parameter measuring risk aversion.

        Parameters
        ----------
        risky_rate : float
            The marginal product of capital, R
        riskfree_rate : float
            The risk free rate of interest, r
        sigma : float
            Standard deviation of technology process
        generation_length : int
            Length of generation, in years

        Returns
        -------
        float
        """

        return math.log(risky_rate / riskfree_rate) / (sigma ** 2) * generation_length


    def _calc_mu(self, b, generation_length, risky_rate, sigma):
        """Calculate mu, the average annual growth rate in the technology process.

        Parameters
        ----------
        b : float
            Capital share of output
        generation_length : int
            Length of generation, in years
        risky_rate : float
            The marginal product of capital, R
        sigma : float
            Standard deviation of technology process

        Returns
        -------
        float
        """

        return -math.log(b) + generation_length * math.log(risky_rate) - (sigma ** 2) / 2


    def simulate(self, transfer=0, transfer_type="fixed"):
        """Simulates economy

        Parameters
        ----------
        transfer : float
            The size of the transfer as a share of capital.
        transfer_type : string
            The type of transfer: "none", "fixed" or "variable"

        Returns
        -------
        None
        """
        self.simulation['capital'][0] = self.parameters.initial_capital

        mean_a = math.exp(self.mu + self.parameters.sigma ** 2 / 2)
        mean_wage = (1 - self.parameters.b) * mean_a

        for period in range(self.parameters.n):
            self.simulation['wage'][period] = (1 - self.parameters.b) * self.a[period]
            self.simulation['risky_rate'][period] = self.parameters.b * self.a[period]
            self.simulation['expected_wages'][period] = mean_wage
            self.simulation['endowment'][period] = self.simulation['expected_wages'][period] if self.with_endowment else 0

            if period > 0:
                self.simulation['transfer_young'][period] = transfer * self.simulation['expected_wages'][period - 1]
            else:
                self.simulation['transfer_young'][period] = transfer * mean_wage

            if transfer_type == 'variable':
                self.simulation['transfer_young'][period] = self.simulation['transfer_young'][period] * self.a[period] / mean_a

            self.simulation['income'][period] = self.simulation['wage'][period] + self.simulation['endowment'][period] - self.simulation['transfer_young'][period]
            savings = self._find_savings(period, mean_wage, transfer, transfer_type)
            self.simulation['consumption_young'][period] = self.simulation['income'][period] - savings

            self.simulation['utility_young'][period], self.simulation['utility_old'][period], self.simulation['utility_total'][period] = \
                self._calc_utilities(period, mean_wage, savings, transfer, transfer_type)

            # Set capital in next period to savings in this one, divided by one plus growth rate
            if period < (self.parameters.n - 1):
                self.simulation['capital'][period + 1] = savings / ((1 + self.growth_rate) ** self.parameters.generation_length)

        return None


    def _find_savings(self, period, mean_wage, transfer, transfer_type):
        """Returns the level of savings consistent with the Euler equation and budget constraint.

        Parameters
        ----------
        period : int (unpacked from args)
            The period of the simulation in which savings is being calculated
KP: Not correct here.
        next_a_values : 1-D array of floats
            The set of possible values of a in the next period
        transfer : float or 1-D array of floats
            The amount that the representative household will receive in their old age from a transfer.

        Returns
        -------
        savings_solution : float
            The level of savings that is consistenet with the budget constraint and Euler equation.
        """
        if transfer == 0:
            return self.parameters.beta * self.simulation['income'][period]
        else:
            savings_guess = self.parameters.beta * self.simulation['income'][period] \
                - (1 - self.parameters.beta) * transfer * mean_wage
            next_a_values = np.exp(self.values * self.parameters.sigma + self.mu)
            transfer_values = transfer * mean_wage
            if transfer_type == 'variable':
                transfer_values = transfer_values * next_a_values / \
                    math.exp(self.mu + self.parameters.sigma ** 2 / 2)

            args = (self, period, next_a_values, transfer_values)

            savings_solution, infodict, ier, mesg = optimize.fsolve(
                self._savings_deviation,
                savings_guess,
                args,
                full_output=True
            )

            if ier != 1:
                raise Exception('Problem with calibrated_beta in ModelLinear._find_savings()')

        return savings_solution


    def _savings_deviation(self, savings, *args):
        """Returns the difference bewteen a candidate level of savings and the levels of savings
        it implies through the Euler equation and the budget constraint. It will equal 0 when the
        fsolve function finds a solution.

        Parameters
        ----------
        period: int (unpacked from args)
            The period of the simulation in which savings is being calculated
        next_a_values: 1-D array of floats
            The set of possible values of a in the next period
        transfer: float or 1-D array of floats
            The amount that the representative household will receive in their old age from a transfer.

        Returns
        ----------
        deviation: float
            Deviation between candidate value of savings and implied savings.
        """
        (self, period, next_a_values, transfer) = args

        right_hand_side_top = np.matmul((self.parameters.b * next_a_values) *
            (self.parameters.b * next_a_values * savings + transfer) ** - self.gamma, self.weights)
        right_hand_side_bottom = np.matmul((self.parameters.b * next_a_values * savings + transfer) ** (1 - self.gamma), self.weights)

        # Euler equations implies consumption in time t = (1 - beta) / beta * right_hand_side_bottom / right_hand_side_top
        # so income - (1 - beta) / beta * right_hand_side_bottom / right_hand_side_top is implied savings
        deviation = self.simulation['income'][period] - (1 - self.parameters.beta) / self.parameters.beta \
            * right_hand_side_bottom / right_hand_side_top - savings

        return deviation.flatten()


    def _calc_utilities(self, period, mean_wage, savings, transfer, transfer_type):
        """Calculate utilities in each generation for all periods.

        Parameters
        ----------
        transfer: float
            The size of the transfer as a share of capital
        transfer_type: string
            The type of transfer: "none", "fixed" or "variable"

        Returns
        ----------
        None
        """
        utility_young = (1 - self.parameters.beta) * math.log(self.simulation['consumption_young'][period])
        next_a_values = np.exp(self.values * self.parameters.sigma + self.mu)
        transfer_values = transfer * mean_wage
        if transfer_type == 'variable':
            transfer_values = transfer_values * next_a_values / \
                math.exp(self.mu + self.parameters.sigma**2 / 2)
        utility_old = self.parameters.beta / (1 - self.gamma) * math.log(np.matmul((1 + self.growth_rate)**self.parameters.generation_length * \
            (self.parameters.b * next_a_values * savings + transfer_values) ** - self.gamma, self.weights))
        utility_total = utility_old + utility_young

        return utility_young, utility_old, utility_total


    def calc_transfer_to_wage_ratio(self, transfer_to_capital_ratio):
        """Converts the transfer as a share of capital into a share of expected wages.

        Parameters
        ----------
        transfer_to_capital_ratio : float
            The size of the transfer as a share of capital.

        Returns
        -------
        transfer_to_wage_ratio:
            The size of the transfer as a share of expected wages
        """

        avg_capital_to_wage_ratio = np.mean(
            self.simulation['capital'][self.parameters.transition_length:] /
            self.simulation['expected_wages'][self.parameters.transition_length:]
        )
        transfer_to_wage_ratio = transfer_to_capital_ratio * avg_capital_to_wage_ratio

        return transfer_to_wage_ratio


    def store_results(self, transfer=0, transfer_type='none'):
        """Stores simulation results in a pandas DataFrame.

        Parameters
        ----------
        transfer : float
            The size of the transfer as a share of capital
        transfer_type : string
            The type of transfer: "none", "fixed" or "variable"

        Returns
        -------
        None
        """

        for chi in self.parameters.chi_values:
            i = len(self.simulation_results.index)

            self.simulation_results.loc[i, 'transfer'] = float(transfer)
            self.simulation_results.loc[i, 'transfer_type'] = transfer_type
            self.simulation_results.loc[i, 'risky_rate'] = self.risky_rate
            self.simulation_results.loc[i, 'riskfree_rate'] = self.riskfree_rate
            self.simulation_results.loc[i, 'growth_rate'] = self.growth_rate
            self.simulation_results.loc[i, 'with_endowment'] = self.with_endowment
            self.simulation_results.loc[i, 'mu'] = self.mu
            self.simulation_results.loc[i, 'chi'] = float(chi)
            self.simulation_results.loc[i, 'gamma'] = self.gamma
            self.simulation_results.loc[i, 'average_capital'] = np.mean(self.simulation['capital'])
            self.simulation_results.loc[i, 'average_wage'] = np.mean(self.simulation['wage'])
            self.simulation_results.loc[i, 'average_consumption_young'] = np.mean(self.simulation['consumption_young'])
            self.simulation_results.loc[i, 'average_risky_rate'] = np.mean(self.simulation['risky_rate'])
            self.simulation_results.loc[i, 'average_income'] = np.mean(self.simulation['income'])
            self.simulation_results.loc[i, 'average_total_utility'] = self._calc_avg_total_utility(
                self.simulation['utility_total'],
                self.parameters.transition_length,
                chi
            )


    def _calc_avg_total_utility(self, total_utility, transition_length, chi):
        """Calculate average total utility over all simulations after transition period.

        Parameters
        ----------
        total_utility : 1-D np.array
            Vector of utilities of length n (number of generations in a simulation)
        transition_length : int
            Assumed number of generations before reaching steady state.
        chi : float
            Aversion to the risk that a generation will be born into a lifetime of low consumption.

        Returns
        -------
        float
            Average total utility
        """

        if chi == 1:
            return np.mean(np.exp(total_utility[transition_length:]))
        else:
            return np.mean(np.exp(total_utility[transition_length:]) ** (1 - chi)) ** (1 / (1 - chi))


    def calc_welfare_effects(self, sim_results):
        """Merge filtered simulation results DataFrames and calculate components of welfare effects:
        crowd_out_effect, risk_shifting_effect, and total_welfare_effect.

        Parameters
        ----------
        sim_results : pd.DataFrame
            DataFrame containing summarized simulation results.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with simulation summary and columns containing welfare effects.
        """

        without_transfer = sim_results[sim_results['transfer_type'] == 'none'].drop(columns=['transfer', 'transfer_type'])
        fixed_transfer = sim_results[sim_results['transfer_type'] == 'fixed'].drop(columns=['transfer_type', 'gamma', 'mu'])
        variable_transfer = sim_results[sim_results['transfer_type'] == 'variable'].drop(columns=['transfer', 'transfer_type', 'gamma', 'mu'])

        merge_cols = ['with_endowment', 'growth_rate', 'risky_rate', 'riskfree_rate', 'chi']
        transfers = fixed_transfer.merge(variable_transfer, on=(merge_cols), suffixes=("_fixed", "_variable"))
        summary = without_transfer.merge(transfers, on=(merge_cols), suffixes=(None, None))

        summary['crowd_out_effect'] = (summary['average_total_utility_variable'] - summary['average_total_utility']) / summary['average_total_utility'] * 100
        summary['risk_shifting_effect'] = (summary['average_total_utility_fixed'] - summary['average_total_utility_variable']) / summary['average_total_utility'] * 100
        summary['total_welfare_effect'] = summary['crowd_out_effect'] + summary['risk_shifting_effect']

        return summary
