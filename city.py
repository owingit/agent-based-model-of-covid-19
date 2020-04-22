import math
import random
import collections
import policy
import numpy as np
import itertools

import networkx as nx


from agent import *


class City:
    def __init__(self, name, x, y, n, edge_proximity, gamma, hpolicy, mpolicy):
        '''Defines an agent, which represents a node in the city-level infection network.

        :param str name: name of the city
        :param int x: width
        :param int y: height
        :param int n: num agents in city
        :param float edge_proximity: edge proximity (proxy for infectivity)
        :param float gamma: experimental gamma denominator
        :param str hpolicy: health policy name
        :param list[str, dict] mpolicy: movement policy name
        '''
        self.gamma = gamma  # gamma naught for covid 19
        self.num_infected = 0
        self.num_removed = 0
        self.num_susceptible = 0
        self.N = n

        self.name = name
        self.hpolicy = hpolicy
        self.mpolicy = mpolicy

        self.width = x
        self.height = y
        self.datafile = '{}x{}x{}parameter_sweep_data.txt'.format(self.width, self.height, self.N)

        self.area = self.width * self.height
        self.agents_per_market = 50
        self.agents_per_transit = 100
        self.agents_per_work = 15
        self.agents_per_home = 3

        self.past_networks = []
        self.network = None
        self.edge_proximity = edge_proximity  # proxy for infectivity
        self.agents = [Agent(i, self) for i in range(0, self.N)]

        self.setup_agent_central_locations()
        #
        # for agent in self.agents:
        #     print('{} has locations work:{}\nhome:{}\nmarket:{}\ntransit:{}\n'.format(agent.name,
        #                                                                               agent.personal_central_locations['work'],
        #                                                                               agent.personal_central_locations['home'],
        #                                                                               agent.personal_central_locations['market'],
        #                                                                               agent.personal_central_locations['transit']
        #                                                                               ))
        self.agent_dict = {v.number: v for v in self.agents}
        self.policy = policy.Policy(self.name, self.agents, self.hpolicy, self.mpolicy)

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

    def remove_overutilized_regions(self, agent_used_regions, used_regions, market_regions,
                                    transit_regions, work_regions, home_regions):
        """If a region has too many points within it, exclude it so that central locations are better distributed."""

        if used_regions['market'].count(agent_used_regions['market']) > self.agents_per_market:
            _region_to_remove = [tup for tup in market_regions if tup[0] == agent_used_regions.get('market')]
            if _region_to_remove:
                market_regions.remove(_region_to_remove[0])

        if used_regions['transit'].count(agent_used_regions['transit']) > self.agents_per_transit:
            _region_to_remove = [tup for tup in transit_regions if tup[0] == agent_used_regions.get('transit')]
            if _region_to_remove:
                transit_regions.remove(_region_to_remove[0])

        if used_regions['work'].count(agent_used_regions['work']) > self.agents_per_work:
            _region_to_remove = [tup for tup in work_regions if tup[0] == agent_used_regions.get('work')]
            if _region_to_remove:
                work_regions.remove(_region_to_remove[0])

        if used_regions['home'].count(agent_used_regions['home']) > self.agents_per_home:
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
        return {
            'susceptible': self.num_susceptible,
            'infected': self.num_infected,
            'removed': self.num_removed,
            'total_IR': self.num_infected + self.num_removed
        }

    def print_states(self):
        print('City: {}\nSusceptible: {}\nInfected: {}\nRemoved: {}\n'.format(
            self.name, self.num_susceptible, self.num_infected, self.num_removed))

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
            agent.set_policy(health_policy=self.hpolicy, movement_policy=self.mpolicy, i=i)
            if i > 0:
                agent.move()
            self.network.add_node(agent)

        # generate edges O(n^2)
        potential_edges = self.find_edge_candidates()

        # add O(|E)
        for edge_number_tuple in potential_edges:
            self.network.add_edge(self.agents[edge_number_tuple[0]], self.agents[edge_number_tuple[1]])

        self.past_networks.append(self.network)

        # infect O(n * |neighbor_set|)
        SI_transition_rates = []
        for agent in self.agents:
            if not agent.has_transitioned_this_timestep():
                if agent.is_infected():
                    SI_transition_rates.append(self.handle_infection(agent))
        beta = sum(SI_transition_rates)

        homes = [agent for agent in self.agents if agent.mode == 'home']
        len_homes = len(homes)
        works = [agent for agent in self.agents if agent.mode == 'work']
        len_works = len(works)
        transits = [agent for agent in self.agents if agent.mode == 'transit']
        len_transits = len(transits)
        markets = [agent for agent in self.agents if agent.mode == 'market']
        len_markets = len(markets)
        if i > 0:
            print('{} stayed home, {} went to work, {} went on the bus, {} went to the market'.format(
                len_homes, len_works, len_transits, len_markets
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
        """What to do when an agent is infected.

        1. Check for susceptible neighbors.
        2. For each susceptible neighbor:
           transmit infection
        3. Recover if t_infected > 1/gamma
        """
        agent.timesteps_infected += 1
        adjacency_list = self.network[agent]
        si_transition_rate = 0
        msg = 'infected_agent {} went to {} and contacted {}'
        if len(adjacency_list) > 0:
            susceptible_neighbors = [neighbor for neighbor in adjacency_list if neighbor.is_susceptible()]
            if len(susceptible_neighbors) > 0:
                si_transition_rate = len(susceptible_neighbors) / self.N
                for neighbor in susceptible_neighbors:
                    if random.random() < si_transition_rate:
                        print(msg.format(agent.name, agent.mode, neighbor.name))
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

        return si_transition_rate
