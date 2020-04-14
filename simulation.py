from city import *
from scipy import special
import numpy as np


def main():
    timesteps = 7
    cities = construct_cities()
    for city in cities:
        city.set_initial_states()
    city_data = {}
    for city in cities:
        city_data[city.name] = []

    for i in range(0, timesteps):
        for city in cities:
            print('Day {}'.format(i))
            city.timestep(i)
            state_dict = city.get_states()
            city_data[city.name].append(state_dict)
            city.print_states()


def construct_cities():
    #  gamma and min/max ro values from https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
    # TODO: different Ro for different cities, based on data?
    gamma = 1.0 / 18.0

    median_R0 = 5.7

    beta = gamma * median_R0

    ws = [50, 70]
    hs = [50, 90]
    ns = [1000, 2000]
    # TODO: choose realistic numbers
    cities = [City('Boulder', ws[0], hs[0], ns[0], beta, gamma),
              City('Denver', ws[1], hs[1], ns[1], beta, gamma),]
              #City('New York', 450, 650, 15000)]
    return cities


if __name__ == "__main__":
    main()