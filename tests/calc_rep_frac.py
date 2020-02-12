"""Calculate fraction of repeaters in detected frbs."""
from bisect import bisect
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from frbpoppy import CosmicPopulation, Survey, LargePopulation, pprint

from convenience import plot_aa_style, rel_path

MAX_DAYS = 4
# Chime started in Aug 2018. Assuming 2/day for one-offs.
# Total of 9 repeaters published on 9 Aug 2019. = ~year
N_CHIME = {'rep': 9, 'one-offs': 365*2, 'time': 3}
SAVE = True
USE_SAVE = False

if USE_SAVE:
    df = pd.read_csv(rel_path(f'plots/rep_frac.csv'))
    days = df.days.values
    fracs = df.fracs.values
else:
    r = CosmicPopulation.simple(1e5, n_days=MAX_DAYS, repeaters=True)
    r.set_dist(z_max=2.)
    r.set_time(model='clustered', r=5.7, k=0.34)
    r.set_lum(model='powerlaw', low=1e40, high=1e45, power=0,
              per_source='different')
    r.set_dm_host(model='gauss', mu=100, sigma=100)
    # In between Cordes review & Petroff review
    r.set_dm_igm(model='ioka', slope=950, sigma=0)
    r.set_dm_mw(model='ne2001')
    r.set_dm(mw=True, igm=True, host=True)

    # Set up survey
    survey = Survey('chime', n_days=MAX_DAYS)
    survey.set_beam(model='chime')

    surv_pop = LargePopulation(r, survey, max_size=1e4).pops[0]

    # See how fraction changes over time
    days = np.linspace(0, MAX_DAYS, MAX_DAYS+1)
    fracs = []
    for day in days:
        t = surv_pop.frbs.time.copy()
        time = np.where(t < day, t, np.nan)
        n_rep = ((~np.isnan(time)).sum(1) > 1).sum()
        n_one_offs = ((~np.isnan(time)).sum(1) == 1).sum()
        frac = n_rep / (n_rep + n_one_offs)
        fracs.append(frac)

    if SAVE:
        # Save data for later plotting if necessary
        df = pd.DataFrame({'days': days, 'fracs': fracs})
        df.to_csv(rel_path(f'plots/rep_frac.csv'))
        surv_pop.save()

# Set up plot style
plot_aa_style(cols=2)
plt.plot(days, fracs)

# Plot CHIME detection line
chime_frac = N_CHIME['rep'] / (N_CHIME['rep'] + N_CHIME['one-offs'])

# Add CHIME detection line
try:
    plt.plot([N_CHIME['time'], N_CHIME['time'], 0],
             [0, chime_frac, chime_frac],
             color='r', linestyle='--', label='chime')
except IndexError:
    pprint(f'CHIME fraction of {chime_frac:.1} not in plotting area')

# Further plot details
plt.xlabel(r'Time (days)')
plt.ylabel(r'$N_{\textrm{repeaters}}/N_{\textrm{detections}}$')
plt.xlim(0, max(days))
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig(rel_path(f'plots/rep_frac.pdf'))
plt.clf()
