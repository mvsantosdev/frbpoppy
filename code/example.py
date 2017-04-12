from do_populate import generate
from do_survey import observe
from do_plot import plot

# Generate FRB population
population = generate(1000,
                lum_dist_pars=[1e40, 1e45, -1.4],
                z_max=1.0,
                pulse=[0.1,10])

# Observe FRB populations
apertif = observe(population, 'APERTIF')
wholesky = observe(population, 'WHOLESKY')
test = observe(population, 'TEST')

# Plot populations
plot(population, apertif, wholesky, test)
