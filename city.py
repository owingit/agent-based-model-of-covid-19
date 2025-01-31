import math
import random
import collections
import policy
import numpy as np
import itertools
import scipy
from scipy import spatial
import matplotlib.pyplot as plt

import networkx as nx
import seaborn as sns

from agent import *


class City:
    def __init__(self, name, x, y, n, edge_proximity, gamma, hpolicy, mpolicy, frequencies_dict):
        '''Defines an agent, which represents a node in the city-level infection network.

        :param str name: name of the city
        :param int x: width
        :param int y: height
        :param int n: num agents in city
        :param float edge_proximity: edge proximity (proxy for infectivity)
        :param float gamma: experimental gamma denominator
        :param str hpolicy: health policy name
        :param list[str, dict] mpolicy: movement policy name
        :param dict frequencies_dict: dictionary of special point frequencies
        '''
        self.POLICIES = None

        self.gamma = gamma  # gamma naught for covid 19
        self.num_infected = 0
        self.num_removed = 0
        self.num_susceptible = 0
        self.num_quarantined = 0
        self.N = n

        self.name = name

        self.width = x
        self.height = y
        self.datafile = '{}x{}x{}parameter_sweep_data.txt'.format(self.width, self.height, self.N)

        self.area = self.width * self.height
        self.agents_per_market = frequencies_dict['market']
        self.agents_per_transit = frequencies_dict['transit']
        self.agents_per_work = frequencies_dict['work']
        self.agents_per_home = frequencies_dict['home']

        self.past_networks = []
        self.network = None
        self.edge_proximity = edge_proximity  # proxy for infectivity
        self.policy = policy.Policy(hpolicy, mpolicy)

        self.agents = [Agent(i, self) for i in range(0, self.N)]

        self.quarantine_center_location=None
        self.quarantine_threshold = 4
        self.quarantine_rate = 0.05
        
        self.setup_agent_central_locations()
        self.agent_dict = {v.number: v for v in self.agents}

    def setup_agent_central_locations(self):
        """Function to initialize central locations for each agent.

        Based on the voronoi diagrams around points chosen by a Poisson point process
        """
        central_locations = self.poisson_point_process(
            intensity=(self.N / self.area) / self.agents_per_market)  # one grocery store for every 50 agents
        transit_hubs = self.poisson_point_process(
            intensity=(self.N / self.area) / self.agents_per_transit)  # one public transit for every 100 agents
        workspaces = self.poisson_point_process(
            intensity=(self.N / self.area) / self.agents_per_work)  # one workplace for every 15 agents
        homes = self.poisson_point_process(intensity=(self.N / self.area) / self.agents_per_home)  #  one home per every 3 agents
        #print(len(set(homes)), len(set(central_locations)), len(set(transit_hubs)), len(set(workspaces)))

        market_regions, transit_regions, work_regions, home_regions = self.setup_voronoi_diagrams(
            markets=central_locations,
            transits=transit_hubs,
            workspaces=workspaces,
            homes=homes)
        #print(len(home_regions), len(market_regions), len(transit_regions), len(work_regions))

        used_regions = {'market': [],
                        'transit': [],
                        'work': [],
                        'home': []
                        }

        for agent in self.agents:
            agent_used_regions = agent.set_and_verify_locations(
                (market_regions, central_locations),
                (transit_regions, transit_hubs),
                (work_regions, workspaces),
                (home_regions, homes)
            )

            used_regions['market'].append(agent_used_regions['market'])
            used_regions['transit'].append(agent_used_regions['transit'])
            used_regions['work'].append(agent_used_regions['work'])
            used_regions['home'].append(agent_used_regions['home'])

            self.remove_overutilized_regions(agent_used_regions, used_regions,
                                             market_regions, transit_regions,
                                             work_regions, home_regions)
        self.define_quarantine_location()

    def remove_overutilized_regions(self, agent_used_regions, used_regions, market_regions,
                                    transit_regions, work_regions, home_regions):
        """If a region has too many points within it, exclude it so that central locations are better distributed."""

        if used_regions['market'].count(agent_used_regions['market']) > self.agents_per_market:
            if market_regions:
                _region_to_remove = [tup for tup in market_regions if tup[0] == agent_used_regions.get('market')]
                if _region_to_remove:
                    market_regions.remove(_region_to_remove[0])

        if used_regions['transit'].count(agent_used_regions['transit']) > self.agents_per_transit:
            if transit_regions:
                _region_to_remove = [tup for tup in transit_regions if tup[0] == agent_used_regions.get('transit')]
                if _region_to_remove:
                    transit_regions.remove(_region_to_remove[0])

        if used_regions['work'].count(agent_used_regions['work']) > self.agents_per_work:
            if work_regions:
                _region_to_remove = [tup for tup in work_regions if tup[0] == agent_used_regions.get('work')]
                if _region_to_remove:
                    work_regions.remove(_region_to_remove[0])

        if used_regions['home'].count(agent_used_regions['home']) > self.agents_per_home:
            if home_regions:
                _region_to_remove = [tup for tup in home_regions if tup[0] == agent_used_regions.get('home')]
                if _region_to_remove:
                    home_regions.remove(_region_to_remove[0])

    def setup_voronoi_diagrams(self, markets, transits, workspaces, homes):
        """
        Creates voronoi diagrams where the centers of each region are the points marked by the poisson point process.

        :param markets: list of market points
        :param transits: list of transit points
        :param workspaces: list of work points
        :param homes: list of home points
        :return: tuple of voronoi regions
        """
        print('Setting up {} fixed locations'.format(self.name))
        modes = ['market', 'transit', 'work', 'home']
        points_list = [markets, transits, workspaces, homes]
        polygons_dict = {mode: [] for mode in modes}
        for index, location_group in enumerate(points_list):
            try:
                location_group = list(location_group)

                vor = Voronoi(location_group)
                vertices = vor.vertices
                regions = vor.regions
                regions.remove([])
                polygons = []
                for i, reg in enumerate(regions):
                    if len(reg) > 3:
                        polygon_vertices = vertices[reg]
                        point_pairs = []
                        for pair in polygon_vertices:
                            point_pair = (pair[0], pair[1])
                            point_pairs.append(point_pair)
                        polygon = Polygon(point_pairs)
                        if i > len(location_group):
                            i = i % len(location_group)
                        polygons.append((i, polygon))
                polygons_dict[modes[index]].extend(polygons)
            except spatial.qhull.QhullError:
                print('Catching Qhull error, defaulting to random network wiring')
                return None, None, None, None
            except IndexError:
                print('Catching Index error, defaulting to random network wiring')
                return None, None, None, None
        return polygons_dict['market'], polygons_dict['transit'], polygons_dict['work'], polygons_dict['home']

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

    def poisson_point_process(self, intensity):
        """Generate central locations based on poisson intensity."""
        num_points = np.random.poisson(intensity * self.area)  # Poisson number of points
        xs = self.width * np.random.uniform(0, 1, num_points)
        ys = self.height * np.random.uniform(0, 1, num_points)
        points = zip(xs, ys)
        return list(points)

    def get_states(self):
        """Returns dict of states.

        :rtype dict(any)
        """
        quarantined = [agent for agent in self.agents if agent.been_quarantined]
        self.num_quarantined = len(quarantined)

        return {
            'susceptible': self.num_susceptible,
            'infected': self.num_infected,
            'removed': self.num_removed,
            'total_IR': self.num_infected + self.num_removed,
            'quarantined': self.num_quarantined
        }

    def print_states(self):
        print('City: {}\nSusceptible: {}\nInfected: {}\nRemoved: {} \nQuarantined : {}'.format(
            self.name, self.num_susceptible, self.num_infected, self.num_removed, self.num_quarantined))

    def view_all_policies(self, policies_dict):
        self.POLICIES = policies_dict

    def view_all_policies(self, policies_dict):
        self.POLICIES = policies_dict

    def timestep(self, i):
        '''One unit of time in a city.

        All cities have the same units of time.
        In a timestep:
            a) all agents in a city move, either within the city or to another city with a certain p
            b) a proximity network is formed
            c) infection spreads with probability gamma
            d) agent trajectories are updated by their distance policy
        '''
        self.network = nx.Graph()

        # move nodes
        # generate nodes O(n)
        for agent in self.agents:
            # TODO: this should probably be a function (set alternative policy)
            if 'essential' in self.policy.movement_policy_name:
                if agent.number % 50 == 0:  # 50 essential workers
                    agent.set_policy(self.policy, i=i)
                else:
                    temp_policy = policy.Policy(self.policy.health_policy, ('preferential_return_stay_at_home',
                                                self.POLICIES['stay_at_home']))
                    agent.set_policy(temp_policy, i=i)
            else:
                agent.set_policy(self.policy, i=i)
            if i > 0:
                if not agent.been_quarantined:
                    agent.move()
            self.network.add_node(agent)

        # generate edges O(n^2)
        potential_edges = self.find_edge_candidates()

        # add O(|E)
        for edge_number_tuple in potential_edges:
            self.network.add_edge(self.agents[edge_number_tuple[0]], self.agents[edge_number_tuple[1]])

        self.past_networks.append(self.network)

        # infect O(n * |neighbor_set|)
        si_transition_rates = []

        for agent in self.agents:
            if not agent.has_transitioned_this_timestep():
                if agent.is_susceptible():
                    si_transition_rates.append(self.handle_infection(agent))
                if agent.is_infected():
                    agent.timesteps_infected += 1
                    quarantine_instance = random.random()
                    if not agent.been_quarantined:
                        if agent.timesteps_infected >= self.quarantine_threshold:
                            if quarantine_instance <= self.quarantine_rate:
                                self.quarantine(agent)
                    self.i_r_transition(agent)

        beta = sum(si_transition_rates)

        homes = [agent for agent in self.agents if agent.mode == 'home']
        len_homes = len(homes)
        works = [agent for agent in self.agents if agent.mode == 'work']
        len_works = len(works)
        transits = [agent for agent in self.agents if agent.mode == 'transit']
        len_transits = len(transits)
        markets = [agent for agent in self.agents if agent.mode == 'market']
        len_markets = len(markets)
        quarantined = [agent for agent in self.agents if agent.been_quarantined]
        len_quarantined = len(quarantined)

        if i > 0:
            print('{} stayed home, {} went to work, {} went on the bus, {} went to the market {} are in quarantine'.format(
                len_homes, len_works, len_transits, len_markets,len_quarantined

            ))
        return beta

    def find_edge_candidates(self):
        """See if a node is close enough to another node to count as an edge.

        Update the health policy and replace the node in self.agents with the modified node.

        :param int i: timestep
        """
        potential_edges = []

        for pair in list(itertools.combinations(self.agents, r=2)):
            d = np.sqrt(
                # euclidean distance
                ((pair[0].positionx - pair[1].positionx) ** 2) + ((pair[0].positiony - pair[1].positiony) ** 2)
            )
            agent_a = pair[0]
            agent_b = pair[1]
            if d <= self.edge_proximity:
                # we know you would be repulsed, so we add you to a data structure here
                potential_edges.append((agent_a.number, agent_b.number))

        return potential_edges

    def handle_infection(self, agent):
        """What to do when an agent is susceptible.

        1. Check for infected neighbors.
        2. For each infected neighbor:
           transmit infection to self with si_transition_rate = contact rate of infection at a timestep
        """
        adjacency_list = self.network[agent]
        si_transition_rate = 0
        infected_neighbors = None
        msg = 'susceptible {} went to {} and became infected'
        if len(adjacency_list) > 0:
            infected_neighbors = [neighbor for neighbor in adjacency_list if neighbor.is_infected()]
            if len(infected_neighbors) > 0:
                si_transition_rate = len(infected_neighbors) / len(adjacency_list)
                if random.random() < si_transition_rate:
                    print(msg.format(agent.name, agent.mode))
                    agent.transition_state('infected')

                    self.num_susceptible -= 1
                    self.num_infected += 1
                    agent.transitioned_this_timestep = True
        return si_transition_rate

    def i_r_transition(self, agent):
        """Recover if t_infected > 1/gamma. If quarantined, send back to home"""
        if agent.timesteps_infected >= (1 / self.gamma):
            print('Transitioning {} to removed'.format(agent.name))
            agent.transition_state('removed')
            self.num_infected -= 1
            self.num_removed += 1
            agent.transitioned_this_timestep = True
            agent.timesteps_infected = 0
            if agent.been_quarantined:
                agent.not_quarantined()
                agent.send_to_home()

    @staticmethod
    def quarantine(agent):
        '''Quarantining an agent and sending the agent to the Q.C.'''
        print('Quarantining {} to Quarantine Center'.format(agent.name))
        agent.has_been_quarantined()
        agent.send_to_quarantine_center()

    def plot_scatter(self, j):
        ''' For visualising the spread of disease as it moves through the city'''
        state_index = 0
        colors = []
        for agent in self.agents:
            if agent.state == 'susceptible':
                colors.append("blue")
            if agent.state == 'infected':
                if agent.been_quarantined:
                    colors.append("black")
                else:
                    colors.append("red")
            if agent.state == 'removed':
                colors.append("green")

        sns.set_style("darkgrid")
        plt.ioff()
        fig = plt.figure()
        for agent in self.agents:
            plt.scatter(agent.positionx,
                        agent.positiony,
                        c=colors[state_index],
                        alpha=0.5)
            state_index += 1
        scatterfile = "plots/{}{}.png".format(self.name, j)
        plt.savefig(scatterfile, dpi=300)
        plt.close(fig)

    def change_proximity(self, epsilon):
        '''Change proxmity radius to simulate Social Distancing.

        :param epsilon: New proximity value
        '''

        self.edge_proximity = epsilon

    def define_quarantine_location(self):
        '''Define location of quarantine center outside the city.'''

        quarantine_x = self.width + 50
        quarantine_y= self.height * 0.5
        self.quarantine_center_location = [quarantine_x, quarantine_y]
