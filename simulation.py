from city import *
from scipy import special
import numpy as np

def main():
    timesteps = 150
    cities = construct_cities()
    for city in cities:
        city.set_initial_states()
    for i in range(0, timesteps):
        for city in cities:
            print('Day {}'.format(i))
            city.timestep(i)
            city.print_states()


def construct_cities():
    #  gamma and min/max ro values from https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
    gamma = 1.0 / 18.0

    median_R0 = 5.7

    beta = gamma * median_R0

    w = 100
    h = 100
    n = 5000
    # TODO: choose realistic numbers
    cities = [City('Boulder', w, h, n, beta, gamma),]
              #City('Denver', 500, 500, 10000),
              #City('New York', 450, 650, 15000)]
    return cities


if __name__ == "__main__":
    main()