import json

import matplotlib.pyplot as plt


class CityGraph:
    def __init__(self, city):
        self.city = city
        self.name = city.name
        self.beta = city.beta
        self.gamma = city.gamma
        self.N = city.N
        self.datafile = city.datafile
        self.xs = []
        self.ys = []
        self.timestep_of_convergence = None
        self.total_infected = 0

    def write_data(self):
        data_line = '{} {}\n'.format(self.beta * (1 / self.gamma), self.total_infected)
        with open(self.datafile, 'a+',) as f:
            f.write(data_line)

    def plot_data(self):
        plot_dict_susceptible = {}
        plot_dict_infected = {}
        plot_dict_removed = {}

        for i, d in enumerate(self.ys):
            plot_dict_susceptible[i] = d['susceptible']
            plot_dict_infected[i] = d['infected']
            plot_dict_removed[i] = d['removed']

        title = "{}. {}x{}, {} agents. Ro: {}".format(self.name,
                                                      self.city.width,
                                                      self.city.height,
                                                      self.N,
                                                      self.beta * 1 / self.gamma)
        plt.title(title)
        plt.plot(list(plot_dict_susceptible.keys()), list(plot_dict_susceptible.values()), label="Susceptible")
        plt.plot(list(plot_dict_infected.keys()), list(plot_dict_infected.values()), label="Infected")
        plt.plot(list(plot_dict_removed.keys()), list(plot_dict_removed.values()), label="Removed")
        plt.xlabel('t')
        plt.ylabel('Num agents in states S,I,R')

        plt.show()
