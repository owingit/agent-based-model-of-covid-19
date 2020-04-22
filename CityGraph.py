import json

import matplotlib.pyplot as plt
from matplotlib import style


class CityGraph:
    def __init__(self, city):
        self.city = city
        self.name = city.name
        self.edge_proximity = city.edge_proximity
        self._beta = None
        self.gamma = city.gamma
        self.N = city.N
        self.datafile = city.datafile
        self.xs = []
        self.ys = []
        self.betas = []
        self.timestep_of_convergence = None
        self.total_infected = 0

    @property
    def beta(self):
        if self._beta:
            return self._beta
        else:
            raise('Beta not set!')

    def set_beta(self, beta):
        self._beta = beta
        self.betas.append(beta)

    def write_data(self):
        data_line = '{} {}\n'.format(self.beta * (1 / self.gamma), self.total_infected)
        with open(self.datafile, 'a+',) as f:
            f.write(data_line)

    def plot_data(self):
        """Plot S, I, R curves over time.

        # TODO: animation?
        """
        plot_dict_susceptible = {}
        plot_dict_infected = {}
        plot_dict_removed = {}

        for i, d in enumerate(self.ys):
            plot_dict_susceptible[i] = d['susceptible']
            plot_dict_infected[i] = d['infected']
            plot_dict_removed[i] = d['removed']

        movement_policy = self.city.policy.movement_policy[1][-1]

        title = "{}. {}x{}, {} agents. {} and {}. Edge proximity: {}".format(self.name,
                                                                                     self.city.width,
                                                                                     self.city.height,
                                                                                     self.N,
                                                                                     movement_policy,
                                                                                     self.city.policy.health_policy,
                                                                                     self.city.edge_proximity)
        style.use('ggplot')
        plt.title(title)
        plt.plot(list(plot_dict_susceptible.keys()), list(plot_dict_susceptible.values()), label="Susceptible")
        plt.plot(list(plot_dict_infected.keys()), list(plot_dict_infected.values()), label="Infected")
        plt.plot(list(plot_dict_removed.keys()), list(plot_dict_removed.values()), label="Removed")
        plt.xlabel('t')
        plt.ylabel('# agents in states S,I,R')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def plot_ro(self):
        plot_dict_betas = {}
        for i, beta in enumerate(self.betas):
            plot_dict_betas[i] = beta * 1 / self.gamma
        title = "R_o over time with edge proximity {}".format(self.edge_proximity)
        style.use('ggplot')
        plt.title(title)
        plt.plot(list(plot_dict_betas.keys()), list(plot_dict_betas.values()), label="Ro")
        plt.xlabel('t')
        plt.ylabel('R_o')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
