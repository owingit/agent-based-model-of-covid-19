from city import *
from CityGraph import *
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys

GAMMAS = np.linspace(1.0, 20.0, num=20)  # infection length (days)
BETAS = np.linspace(1.0, 100.0, num=100) / 100
DO_PARAMETER_SWEEP = False
# COVID_Ro = 5.7  # gamma and min/max ro values from https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
COVID_Ro = 3.8
COVID_Gamma = 18.0
COVID_Beta = COVID_Ro / COVID_Gamma


def main():
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    if DO_PARAMETER_SWEEP:
        for beta in BETAS:
            for gamma in GAMMAS:
                setup_and_run(timesteps, beta, gamma)
    else:
        setup_and_run(timesteps, COVID_Beta, COVID_Gamma)


def setup_and_run(timesteps, beta, gamma):
    """Initialize the simulation.

    :param timesteps: number of timesteps to run the simulation
    :param beta: beta parameter (proxy for infection rate)
    :param gamma: gamma parameter (proxy for recovery rate)
    """
    cities = construct_cities(beta, gamma, timesteps)

    for city in cities:
        city.set_initial_states()
    city_graphs = []
    for city in cities:
        city_graphs.append(CityGraph(city))

    for i in range(0, timesteps):
        for city, city_graph in zip(cities, city_graphs):
            city_graph.xs.append(i)
            print('Day {} - {}'.format(i, city.name))
            city.timestep(i)
            state_dict = city.get_states()
            if state_dict['total_IR'] == city.N:
                if not city_graph.timestep_of_convergence:
                    city_graph.timestep_of_convergence = i

            city_graph.ys.append(state_dict)
            city.print_states()

    for cg in city_graphs:
        cg.total_infected = cg.ys[len(cg.xs)-1]['total_IR']
        cg.plot_data()
        #cg.write_data()


def construct_cities(beta, gamma_denom, timesteps):
    """Initialize cities with different policies, beta, gamma values. Right now they're all the same size/population

    :param float beta: experimental beta value
    :param float gamma_denom: gamma denominator
    :param int timesteps
    :returns: list of city objects"""
    # TODO: different Ro for different cities, based on data?
    gamma = 1.0 / gamma_denom

    intent = 'tight'
    location_policies_dict_a = construct_location_policies_dict(intent, timesteps)
    mpolicy_a = ['preferential_return', location_policies_dict_a]
    intent = 'lax'
    location_policies_dict_b = construct_location_policies_dict(intent, timesteps)
    mpolicy_b = ['preferential_return', location_policies_dict_b]
    intent = 'normal'
    location_policies_dict_c = construct_location_policies_dict(intent, timesteps)
    mpolicy_c = ['preferential_return', location_policies_dict_c]

    ws = [100, 100]
    hs = [100, 100]
    ns = [1500, 3000]
    hpolicy_a = 'social_distancing'
    hpolicy_b = 'normal'
    cities = [City('Boulder', ws[0], hs[0], ns[0], beta, gamma, hpolicy_a, mpolicy_a),
              City('Denver', ws[1], hs[1], ns[0], beta, gamma, hpolicy_b, mpolicy_b),
              City('Mixopolis', ws[1], hs[1], ns[0], beta, gamma, hpolicy_b, mpolicy_c)]
              # City('Fort Collins', ws[0], hs[0], ns[0], beta, gamma, hpolicy_a, mpolicy_b),
              # City('Colorado Springs', ws[1], hs[1], ns[0], beta, gamma, hpolicy_b, mpolicy_a),
              # City('DenserBoulder', ws[0], hs[0], ns[1], beta, gamma, hpolicy_a, mpolicy_a),
              # City('DenserDenver', ws[1], hs[1], ns[1], beta, gamma, hpolicy_b, mpolicy_b),
              # City('DenserFort Collins', ws[0], hs[0], ns[1], beta, gamma, hpolicy_a, mpolicy_b),
              # City('DenserColorado Springs', ws[1], hs[1], ns[1], beta, gamma, hpolicy_b, mpolicy_a)]
    return cities


def construct_location_policies_dict(intent, timesteps):
    """Make policy different at each timestep."""
    location_policies = {
        'lax': {'home': 0.5, 'work': 0.3, 'market': 0.1, 'transit': 0.1},
        'tight': {'home': 0.9, 'work': 0.05, 'market': 0.03, 'transit': 0.02},
        'normal': {'home': 0.25, 'work': 0.25, 'market': 0.25, 'transit': 0.25}
    }
    location_policies_dict = {}
    for i in range(0, timesteps):
        if i < 10:
            location_policies_dict[i] = location_policies['normal']
        else:
            location_policies_dict[i] = location_policies[intent]
    return location_policies_dict


if __name__ == "__main__":
    main()