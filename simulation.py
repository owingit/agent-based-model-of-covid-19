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
LOCATION_POLICIES = {
    'lax': {'home': 0.3, 'work': 0.3, 'market': 0.1, 'transit': 0.3},
    'tight': {'home': 0.8, 'work': 0.05, 'market': 0.1, 'transit': 0.05},
    'even': {'home': 0.25, 'work': 0.25, 'market': 0.25, 'transit': 0.25},
    'stay_at_home': {'home': 0.9, 'work': 0.00, 'market': 0.05, 'transit': 0.05},
    'essential_worker': {'home': 0.3, 'work': 0.5, 'market': 0.05, 'transit': 0.15},
    'lockdown': {'home': 0.95, 'work': 0.00, 'market': 0.04, 'transit': 0.01},
}
POLICIES = dict()


def main():
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    if DO_PARAMETER_SWEEP:
        for edge_proximity in EDGE_PROXIMITIES:
            for gamma in GAMMAS:
                setup_and_run(timesteps, edge_proximity, gamma)
    else:
        setup_and_run(timesteps, 0.2, COVID_Gamma)


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
            beta = city.timestep(i) / city.N
            city_graph.set_beta(beta)
            state_dict = city.get_states()
            if state_dict['total_IR'] == city.N:
                if not city_graph.timestep_of_convergence:
                    city_graph.timestep_of_convergence = i

            city_graph.ys.append(state_dict)
            city.print_states()
        finished_cities = []
        for city in cities:
            if city.num_infected == 0:
                finished_cities.append(city)
        if len(finished_cities) == len(cities):
            print('All agents are free of infection.')
            break

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
    mpolicy_a = ['preferential_return_tight', location_policies_dict_a]
    intent = 'stay_at_home'
    location_policies_dict_b = construct_location_policies_dict(intent, timesteps)
    mpolicy_b = ['preferential_return_stay_at_home', location_policies_dict_b]
    intent = 'lockdown'
    location_policies_dict_c = construct_location_policies_dict(intent, timesteps)
    mpolicy_c = ['preferential_return_lockdown', location_policies_dict_c]
    intent = 'lax'
    location_policies_dict_d = construct_location_policies_dict(intent, timesteps)
    mpolicy_d = ['preferential_return', location_policies_dict_d]
    frequencies_dict_a = {'market': 50, 'transit': 200, 'work': 20, 'home': 2}

    ws = [200, 300, 50]
    hs = [200, 300, 50]
    ns = [1500, 5000, 600]
    hpolicy_a = 'social_distancing'
    hpolicy_b = 'normal'
    cities = [#City(name='Boulder', x=ws[1], y=hs[1], n=ns[1], edge_proximity=edge_proximity,
    #                gamma=gamma, hpolicy=hpolicy_a, mpolicy=mpolicy_a),
    #           City(name='Denver', x=ws[1], y=hs[1], n=ns[1], edge_proximity=edge_proximity,
    #                gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_b),
    #           City(name='Quarantinopolis', x=ws[1], y=hs[1], n=ns[1], edge_proximity=edge_proximity,
    #                gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_c),
    #           City(name='EssentialWorkerOpolis', x=ws[1], y=hs[1], n=ns[1], edge_proximity=edge_proximity,
    #                gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_d),
              City('Fort Collins', ws[2], hs[2], ns[2], edge_proximity, gamma, hpolicy_a, mpolicy_b,
                   frequencies_dict_a),]
              # City('Colorado Springs', ws[1], hs[1], ns[0], beta, gamma, hpolicy_b, mpolicy_a),
              # City('DenserBoulder', ws[0], hs[0], ns[1], beta, gamma, hpolicy_a, mpolicy_a),
              # City('DenserDenver', ws[1], hs[1], ns[1], beta, gamma, hpolicy_b, mpolicy_b),
              # City('DenserFort Collins', ws[0], hs[0], ns[1], beta, gamma, hpolicy_a, mpolicy_b),
              # City('DenserColorado Springs', ws[1], hs[1], ns[1], beta, gamma, hpolicy_b, mpolicy_a)]
    for city in cities:
        city.view_all_policies(POLICIES)
    return cities


def construct_location_policies_dict(intent, timesteps):
    """Make policy different at each timestep."""
    location_policies_dict = {}
    for i in range(0, timesteps):
        if i < 10:
             location_policies_dict[i] = LOCATION_POLICIES['lax']
        else:
            location_policies_dict[i] = LOCATION_POLICIES[intent]
    POLICIES[intent] = location_policies_dict
    return location_policies_dict


if __name__ == "__main__":
    main()
