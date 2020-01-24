"""Functions to generate Dispersion measure distributions."""
import numpy as np
import frbpoppy.gen_dists as gd


def ioka(z=0, slope=950, sigma=None, spread_func=np.random.normal):
    """Calculate the contribution of the igm to the dispersion measure.

    Follows Ioka (2003) and Inoue (2004), with default slope value falling
    inbetween the Cordes and Petroff reviews

    Args:
        z (array): Redshifts.
        slope (float): Slope of the DM-z relationship.
        sigma (float): Spread around the DM-z relationship.
        spread (function): Spread function option. Choice from
            ('np.random.normal', 'np.random.lognormal')

    Returns:
        dm_igm (array): Dispersion measure of intergalactic medium [pc/cm^3]

    """
    if sigma is None:
        sigma = 0.2*slope*z
    if spread_func.__name__ not in ('normal', 'lognormal'):
        raise ValueError('spread_func input not recognised')
    return spread_func(slope*z, sigma).astype(np.float32)


def gauss(mu=100, sigma=200, n_srcs=1, z=0):
    """Generate dm host contributions similar to Tendulkar.

    Args:
        mu (float): Mean DM [pc/cm^3].
        sigma (float): Standard deviation DM [pc/cm^3].
        n_srcs (int): Number of sources for which to generate values.
        z (int): Redshift of sources.

    Returns:
        array: DM host [pc/cm^3]

    """
    dm_host = gd.trunc_norm(mu, sigma, n_srcs).astype(np.float32)
    return dm_host / (1 + z)


def lognormal(mu=100, sigma=200, n_srcs=1, z=0):
    """Generate a log normal dm host distribution.

    Args:
        mu (float): Mean DM [pc/cm^3].
        sigma (float): Standard deviation DM [pc/cm^3].
        n_srcs (int): Number of sources for which to generate values.
        z (int): Redshift of sources.

    Returns:
        array: DM host [pc/cm^3]

    """
    dm_host = np.random.lognormal(mu, sigma, n_srcs).astype(np.float32)
    return dm_host / (1 + z)