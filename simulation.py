from city import *
from CityGraph import *
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys

GAMMAS = np.linspace(1.0, 20.0, num=20)  # infection length (days)
EDGE_PROXIMITIES = np.linspace(0.01, 1.0, num=100)  # proxy for infectivity
DO_PARAMETER_SWEEP = False
COVID_Gamma = 18.0


def main():
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    if DO_PARAMETER_SWEEP:
        for edge_proximity in EDGE_PROXIMITIES:
            for gamma in GAMMAS:
                setup_and_run(timesteps, edge_proximity, gamma)
    else:
        setup_and_run(timesteps, 0.001, COVID_Gamma)


def setup_and_run(timesteps, edge_proximity, gamma):
    """Initialize the simulation.

    :param timesteps: number of timesteps to run the simulation
    :param edge_proximity: edge proximity parameter (proxy for infection rate)
    :param gamma: gamma parameter (proxy for recovery rate)
    """
    cities = construct_cities(edge_proximity, gamma, timesteps)
    for city in cities:
        city.set_initial_states()
    city_graphs = []
    for city in cities:
        city_graphs.append(CityGraph(city))

    for i in range(0, timesteps):
        for city, city_graph in zip(cities, city_graphs):
            city_graph.xs.append(i)
            print('Day {} - {}'.format(i, city.name))
            beta = city.timestep(i)
            city_graph.set_beta(beta)
            state_dict = city.get_states()
            if state_dict['total_IR'] == city.N:
                if not city_graph.timestep_of_convergence:
                    city_graph.timestep_of_convergence = i

            city_graph.ys.append(state_dict)
            city.print_states()

    for cg in city_graphs:
        cg.total_infected = cg.ys[len(cg.xs)-1]['total_IR']
        cg.plot_data()
        cg.plot_ro()
        #cg.write_data()



def construct_cities(edge_proximity, gamma_denom, timesteps):
    """Initialize cities with different policies, beta, gamma values. Right now they're all the same size/population

    :param float edge_proximity: experimental edge_proximity value
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
    intent = 'lockdown'
    location_policies_dict_c = construct_location_policies_dict(intent, timesteps)
    mpolicy_c = ['preferential_return', location_policies_dict_c]

    ws = [200, 200]
    hs = [200, 200]
    ns = [1500, 3000]
    hpolicy_a = 'social_distancing'
    hpolicy_b = 'normal'
    cities = [City(name='Boulder', x=ws[0], y=hs[0], n=ns[0], edge_proximity=edge_proximity,
                   gamma=gamma, hpolicy=hpolicy_a, mpolicy=mpolicy_a),
              City(name='Denver', x=ws[1], y=hs[1], n=ns[0], edge_proximity=edge_proximity,
                   gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_b),
              City(name='Mixopolis', x=ws[1], y=hs[1], n=ns[0], edge_proximity=edge_proximity,
                   gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_c)]
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
        'lax': {'home': 0.3, 'work': 0.3, 'market': 0.1, 'transit': 0.3},
        'tight': {'home': 0.9, 'work': 0.05, 'market': 0.03, 'transit': 0.02},
        'normal': {'home': 0.25, 'work': 0.25, 'market': 0.25, 'transit': 0.25},
        'lockdown': {'home': 0.99, 'work': 0.00, 'market': 0.01, 'transit': 0.00},
    }
    location_policies_dict = {}
    for i in range(0, timesteps):
        if i < 3:
            location_policies_dict[i] = location_policies['lax']
        location_policies_dict[i] = location_policies[intent]
    return location_policies_dict


if __name__ == "__main__":
    main()