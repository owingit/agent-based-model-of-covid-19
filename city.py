import math
import random
import numpy as np
import itertools

import networkx as nx


from agent import *


class City:
    def __init__(self, name, x, y, n, beta, gamma):
        self.beta = beta  # beta naught for covid 19
        self.gamma = gamma  # gamma naught for covid 19
        self.name = name
        self.policy = None
        self.num_infected = 0
        self.num_removed = 0
        self.num_susceptible = 0
        self.N = n
        self.width = x
        self.height = y
        self.agents = [Agent(i, self, self.beta, self.gamma) for i in range(0, self.N)]
        self.past_networks = []
        self.network = None
        self.edge_proximity = 3.0

    def set_policy(self, policy):
        self.policy = policy

    def print_width(self):
        print('{} is {} units wide'.format(self.name, self.width))

    def print_height(self):
        print('{} is {} units tall'.format(self.name, self.height))

    def print_agents(self):
        for agent in self.agents:
            print('{} is in state {}'.format(agent.name, agent.state))

    def set_initial_states(self):
        patient_zero = random.choice(self.agents)
        print('Patient zero in {} is {}'.format(self.name, patient_zero.name))
        patient_zero.transition_state('infected')
        for agent in self.agents:
            if agent.state == 'susceptible':
                self.num_susceptible += 1
            if agent.state == 'infected':
                self.num_infected += 1
            if agent.state == 'removed':
                self.num_removed += 1

    def print_states(self):
        print('City: {}\nSusceptible: {}\nInfected: {}\nRemoved: {}\n'.format(self.name, self.num_susceptible, self.num_infected, self.num_removed))

    def timestep(self, i):
        '''One unit of time in a city.

        All cities have the same units of time.
        In a timestep:
            a) all agents in a city move, either within the city or to another city with a certain p
            b) a proximity network is formed
            c) infection spreads with probability gamma
        :return:
        '''
        # S_I_transition_rate = self.beta * self.num_infected / self.N
        self.network = nx.Graph()
        # generate nodes O(n)
        for agent in self.agents:
            agent.move()
            self.network.add_node(agent)

        # generate edges O(n^2)
        for pair in list(itertools.combinations(self.agents, r=2)):
            d = np.sqrt(((pair[0].positionx - pair[1].positionx) ** 2) + ((pair[0].positiony - pair[1].positiony) ** 2))
            if d <= self.edge_proximity:
                self.network.add_edge(pair[0], pair[1])

        self.past_networks.append(self.network)

        # infect O(n * |neighbor_set|)
        for agent in self.agents:
            if not agent.has_transitioned_this_timestep():
                if agent.is_infected():
                    agent.timesteps_infected += 1
                    adjacency_list = self.network[agent]
                    if len(adjacency_list) > 0:
                        susceptible_neighbors = [neighbor for neighbor in adjacency_list if neighbor.is_susceptible()]
                        if len(susceptible_neighbors) > 0:
                            S_I_transition_rate = self.beta / len(susceptible_neighbors)
                            print(S_I_transition_rate)  # found by solving beta = alpha * p, where alpha is the contact rate based on the ABM network
                            for neighbor in susceptible_neighbors:
                                if random.random() <= S_I_transition_rate:
                                    print('Transitioning {} to infected'.format(neighbor.name))
                                    self.agents[neighbor.number].transition_state('infected')
                                    self.num_susceptible -= 1
                                    self.num_infected += 1
                                    self.agents[neighbor.number].transitioned_this_timestep = True

                    if agent.timesteps_infected >= (1 / self.gamma):
                        print('Transitioning {} to removed'.format(agent.name))
                        agent.transition_state('removed')
                        self.num_infected -= 1
                        self.num_removed += 1
                        agent.transitioned_this_timestep = True
                        agent.timesteps_infected = 0




