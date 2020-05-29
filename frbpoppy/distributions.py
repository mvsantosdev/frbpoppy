"""Define distributions from which to get random numbers."""
import numpy as np
import math
from scipy.stats import truncnorm
import frbpoppy.precalc as pc

from scipy.special import gamma, gammainc, gammaincinv

def schechter(low, high, power, shape=1):
    """
    Return random variables distributed according to Schechter luminosity function.

    p(x) = x^power exp(-x) dx / k
    
    k: Normalization factor.
    k = Gamma(power + 1, )

    Args:
        low (float): Lower limit of distribution
        high (float): Higher limit of distribution
        power (float): Power of Schechter luminosity function (\alpha).
        shape (int/tuple): Shape of array to be generated. Can also be a int.

    Returns:
        array: Random variable picked from Schechter luminosity function.

    """
    if low > high:
        low, high = high, low

    if isinstance(shape, tuple):
        n_gen = np.prod(shape)
    else:
        n_gen = shape
        
    power1 = power + 1

    gamma_low = gammainc(power1, low)
    gamma_high = gammainc(power1, high)
    
    u = np.random.random(n_gen)
    
    arg = gamma_low + (gamma_high - gamma_low)*u
    
    pl = gammaincinv(power1, arg)

    if isinstance(shape, tuple):
        return pl.reshape(shape)

    return pl

def powerlaw(low, high, power, shape=1):
    """
    Return random variables distributed according to power law.

    The power law distribution power is simply the power, not including a minus
    sign (P scales with x^n with n the power). A flat powerlaw can therefore
    be created by taking setting power to zero.

    Args:
        low (float): Lower limit of distribution
        high (float): Higher limit of distribution
        power (float): Power of power law distribution
        shape (int/tuple): Shape of array to be generated. Can also be a int.

    Returns:
        array: Random variable picked from power law distribution

    """
    if low > high:
        low, high = high, low

    if power == 0 or low == high:
        return 10**np.random.uniform(np.log10(low), np.log10(high), shape)
    
    n_gen = np.prod(shape) if isinstance(shape, tuple) else shape
    
    u = np.random.random(n_gen)
    
    lown = low**power
    highn = high**power
    
    pl = (lown + (highn - lown)*u)**(1/power)

    if isinstance(shape, tuple):
        
        return pl.reshape(shape)

    return pl


def trunc_norm(mu, sigma, n_gen=1, lower=0, upper=np.inf):
    """Draw from a truncated normal distribution.

    Args:
        mu (number): Mu
        sigma (number): Sigma
        n_gen (number): Number to generate
        lower (number): Lower limit
        upper (number): Upper limit

    Returns:
        array: Numpy of required length

    """
    if sigma == 0:
        return np.full(n_gen, mu)
    left = (lower-mu)/sigma
    right = (upper-mu)/sigma
    d = truncnorm.rvs(left, right, loc=mu, scale=sigma, size=n_gen)
    return d


def oppermann_pen():
    """Following Oppermann & Pen (2017), simulate repeat times.
    Returns:
        ts (list): List of burst times
    """
    # TODO Numpy-ify
    r = 5.7
    k = 0.34

    ts = []
    t_tot = 0.5  # Assuming a maximum of 12 hours on one spot
    t_sum = 0.0

    # Get time of bursts
    while t_sum < t_tot:
        t = r*np.random.weibull(k)
        t_sum += t
        ts.append(t_sum)

    # Convert to seconds
    ts = [t*86400 for t in ts[:-1]]

    return ts
