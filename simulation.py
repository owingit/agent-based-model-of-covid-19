from city import *
from CityGraph import *
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

GAMMAS = np.linspace(1.0, 20.0, num=20)  # infection length (days)
BETAS = np.linspace(1.0, 100.0, num=100) / 100
DO_PARAMETER_SWEEP = False
COVID_Ro = 5.7  # gamma and min/max ro values from https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
COVID_Gamma = 18.0
COVID_Beta = COVID_Ro / COVID_Gamma


def main():
    timesteps = 100
    if DO_PARAMETER_SWEEP:
        for beta in BETAS:
            for gamma in GAMMAS:
                setup_and_run(timesteps, beta, gamma)
    else:
        setup_and_run(timesteps, COVID_Beta, COVID_Gamma)


def setup_and_run(timesteps, beta, gamma):
    cities = construct_cities(beta, gamma)

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
        cg.write_data()


def construct_cities(beta, gamma_denom):
    # TODO: different Ro for different cities, based on data?
    gamma = 1.0 / gamma_denom

    ws = [50, 70]
    hs = [50, 90]
    ns = [1000, 2000]
    # TODO: choose realistic numbers
    cities = [City('Boulder', ws[0], hs[0], ns[0], beta, gamma),
              City('Denver', ws[1], hs[1], ns[1], beta, gamma)]
              #City('New York', 450, 650, 15000)]
    return cities


if __name__ == "__main__":
    main()