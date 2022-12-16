from collections import namedtuple
import math
import yaml
import numpy as np
import pandas as pd


def read_parameters(file):
    """Read in fixed model parameters from specified yml file. Returns a namedtuple data structure.

    Parameters
    ----------
    file : str
        Full path to parameters.yml file

    Returns
    -------
    params : namedtuple
    """

    with open(file) as param_file:
        parameters = yaml.load(param_file, Loader=yaml.FullLoader)

    # Need to convert string value for b (1/3) to float value
    parameters['b'] = eval(parameters['b'])

    Parameters = namedtuple('Parameters', parameters.keys())
    params = Parameters(*parameters.values())

    return params


def calc_expected_wages(b, expected_a, risky_rate):
    """Calculate expected wages when the economy is in the steady state.

    Parameters
    ----------
    b : float
        The capital share of output in the Cobb-Douglas production function
    expected_a : float
        The expected value of technology shocks (a) in time t+1, at time t
    risky_rate : float
        The marginal product of capital, R

    Returns
    ----------
    expected_wages : float
        Expected wages in the steady state
    """

    expected_wages = math.exp(
         math.log(1 - b)
       + (b / (1 - b)) * math.log(b)
       + (1 / (1 - b)) * math.log(expected_a)
       + (b / (b - 1)) * math.log(risky_rate)
     )

    return expected_wages


def calc_implied_capital(b, expected_a, risky_rate):
    """Calculate the implied level of capital.

    Calculates the implied level of capital given the risky
    rate (aka marginal product of capital)

    Parameters
    ----------
    b : float
        The capital share of output in the Cobb-Douglas production function
    expected_a : float
        The expected value of technology shocks (a) in time t+1, at time t
    risky_rate : float
        The marginal product of capital, R

    Returns
    -------
    implied_capital : float
        Implied level of capital.
    """

    implied_capital = (risky_rate / expected_a / b) ** (1 / (b - 1))

    return implied_capital


def make_a(mu, sigma, epsilon, rho=0.0):
    """
    Make the time series vector of stochastic values of A, the level of technological productivity
    in the economy

    Parameters
    ----------
    mu: float
        Mean of technology process
    sigma : float
        Standard deviation of technology process
    epsilon : 1-D ndarray
        Array of standard normal shocks in the technology process
    rho : float
        Level of persistence (autocorrelation) in technology process
    """
    n = epsilon.size
    a = np.zeros((n, 1))
    a[0] = np.exp(mu + sigma**2 / 2 / (1 - rho **2))
    for i in range(1, n):
        a[i] = math.exp(mu + rho * (math.log(a[i-1]) - mu) + np.random.normal(0, sigma))
    return a


def calc_transfer_old(transfer, expected_wages, transfer_type, values, sigma):
    """Calculate the transfer that the representative household will receive when old.

    This function returns either a zero in the case that there is no transfer or
    a matrix of nsim * n of transfers, where element i, j is the transfer
    in the ith simulation under the jth value from Gauss-Hermite quadrature.
    If all transfers are 0 then the function outputs a scalar equal to 0.

    Parameters
    ----------
    transfer : float
        The share of steady state wages that is transferred on averge (in case of a variable transfer) or
        with certainty (in the case of a fixed transfer)
    expected_wages : float
        Expected wages in the steady state under the expected value of a
    transfer_type : string
        Set to "fixed" if transfer is know with certainty when young, or "variable" when transfer is directly
        proportional to the value of a at time t+1
    values : 1-D ndarray
        The set possible levels of the variable "a" from the Gauss Hermite quadrature
    sigma : float
        Standard deviation of technology process

    Returns
    -------
    transfer_values : float or 1-D ndarray
        The value of the fixed transfer, or in the case of a variable transfer an array of values
        corresponding to the values of a in "values"
    """
    if (np.array(transfer == 0).all()):
        return 0
    else:
        n = values.shape[1]
        nsim = expected_wages.size
        transfer_values = transfer * np.matmul(expected_wages, np.ones((1, n)))

        if transfer_type == 'fixed':
            return transfer_values
        else:
            transfer_values *= np.exp(np.matmul(np.ones((nsim, 1)), values)) / np.exp(sigma ** 2 / 2)
            return transfer_values
