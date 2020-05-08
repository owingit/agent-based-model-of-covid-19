import json

import matplotlib.pyplot as plt
from matplotlib import style
import time

import seaborn as sns


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

    def plot_data(self, avgs=None):
        """Plot S, I, R curves over time.

        # TODO: animation?
        """
        add_on = ''
        if avgs:
            add_on = 'avg_vals (n={})'.format(avgs)
        plot_dict_susceptible = {}
        plot_dict_infected = {}
        plot_dict_removed = {}
        plot_dict_quarantined = {}
        for i, d in enumerate(self.ys):
            plot_dict_susceptible[i] = d['susceptible']
            plot_dict_infected[i] = d['infected']
            plot_dict_removed[i] = d['removed']
            plot_dict_quarantined[i] = d['quarantined'] 

        movement_policy_probs = self.city.policy.movement_probabilities
        movement_policy = movement_policy_probs[len(self.xs)-1]

        title = "{}. {}x{}, {} agents.\n Quarantine Rate: {} Quarantine Threshold : {} Days".format(
            self.name+add_on,
            self.city.width,
            self.city.height,
            self.N,
            self.city.quarantine_rate,
            self.city.quarantine_threshold
        )
        subtitle = '$p_H$={} , $p_W$={} , $p_M$={} ,$p_T$={} , {} ,Edge Proximity = {}'.format(
            movement_policy['home'],
            movement_policy['work'],
            movement_policy['market'],
            movement_policy['transit'],
            self.city.policy.health_policy,
            self.city.edge_proximity
        )

        sns.set_style("darkgrid")
        fig1, ax = plt.subplots(figsize=(10, 6))
        ax.text(x=0.5, y=1.1, s=title, fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
        ax.text(x=0.5, y=1.05, s=subtitle, fontsize=10, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)

        plt.plot(list(plot_dict_susceptible.keys()), list(plot_dict_susceptible.values()), "b-",
                 label="Susceptible")
        plt.plot(list(plot_dict_infected.keys()), list(plot_dict_infected.values()), "r-",
                 label="Infected")
        plt.plot(list(plot_dict_removed.keys()), list(plot_dict_removed.values()), "g-",
                 label="Removed")
        plt.plot(list(plot_dict_quarantined.keys()), list(plot_dict_quarantined.values()), "k--", linewidth=1,
                 label="Quarantined")

        plt.xlabel('time')
        plt.ylabel('Number of Agents')
        plt.legend(loc='best')
        plotname = "SIR-" + self.name+add_on + "{}".format(self.city.quarantine_rate) + "{}".format(time.strftime("%Y%m%d-%H%M%S")) + ".png"
        plt.savefig('data/{}'.format(plotname), dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close(fig1)
        i_max = max(plot_dict_infected.values())
        return i_max

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
        plotname = "Ro over time in " + self.name + "{}".format(self.city.quarantine_rate) + "{}".format(time.strftime("%Y%m%d-%H%M%S")) + ".png"
        plt.savefig('data/{}'.format(plotname), dpi=300, bbox_inches='tight')
        # plt.show()
