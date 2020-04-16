import math
import random
import numpy as np


class Agent:
    def __init__(self, i, City, beta, gamma):
        '''Defines an agent, which represents a node in the city-level infection network.

        :param int i: num
        '''
        # attributes
        self.infection_beta = beta
        self.infection_gamma = gamma

        self.susceptible = True
        self.infected = False
        self.removed = False

        self.timesteps_infected = 0
        self.city = City

        self.positionx = None
        self.positiony = None
        self.prior_x_position = 0
        self.prior_y_position = 0
        self.prior_direction = 0
        self.direction = 0

        self.theta_star = np.linspace(-(math.pi / 2), (math.pi / 2), 100)
        self.movement_angle_at_current_timestep = self.theta_star[random.randint(0, 99)]
        self.name = "Agent #{}".format(i)
        self.number = i
        self.velocity = 1.0
        self.transitioned_this_timestep = False

        self.initialize_position_and_direction_and_state()

    def initialize_position_and_direction_and_state(self):
        self.positionx = random.random() * self.city.width
        self.positiony = random.random() * self.city.height
        self.direction = self.theta_star[random.randint(0, 99)]
        self.prior_x_position = self.positionx
        self.prior_y_position = self.positiony
        self.prior_direction = self.direction

    def move(self, movement_mode='2d_random_walk'):
        '''Move the agent.'''
        if movement_mode == '2d_random_walk':
            self.twod_random_walk()

        if movement_mode == 'preferential_return':
            self.preferential_return()

        self.recalculate_positions_based_on_edges(self.city)
        self.transitioned_this_timestep = False

    def twod_random_walk(self):
        '''2-d correlated random walk.'''
        self.movement_angle_at_current_timestep = self.theta_star[random.randint(0, 99)]
        self.direction = self.prior_direction + self.movement_angle_at_current_timestep

        #  normal movement, constrained by city boundaries
        self.positionx = self.prior_x_position + (self.velocity * math.cos(self.direction))
        self.positiony = self.prior_y_position + (self.velocity * math.sin(self.direction))

    def preferential_return(self):
        '''Preferential return movement model.'''
        # TODO

    def recalculate_positions_based_on_edges(self, city):
        '''Adjust the positions of an agent based on the city's boundaries.

        :param city:
        :return: tuple(float, float, float) positionx, positiony, theta: x and y coordinates, updated
        '''
        x_modified = False
        y_modified = False
        if self.positionx >= city.width:
            self.positionx = self.positionx - self.velocity
            x_modified = True

        if self.positionx < 0:
            self.positionx = self.positionx * -1 * self.velocity
            x_modified = True

        if self.positiony >= city.height:
            self.positiony = self.positiony - self.velocity
            y_modified = True

        if self.positiony < 0:
            self.positiony = self.positiony * -1 * self.velocity
            y_modified = True

        if x_modified:
            self.direction = math.pi - self.direction

        if y_modified:
            self.direction = self.direction * -1

    def transition_state(self, target_state):
        if target_state == 'removed':
            self.removed = True
            self.infected = False
            self.susceptible = False
        if target_state == 'susceptible':
            self.removed = False
            self.infected = False
            self.susceptible = True
        if target_state == 'infected':
            self.removed = False
            self.infected = True
            self.susceptible = False

    @property
    def get_city(self):
        return self.city

    @property
    def state(self):
        if self.susceptible:
            return 'susceptible'
        elif self.infected:
            return 'infected'
        else:
            return 'removed'

    def is_infected(self):
        return self.infected

    def is_susceptible(self):
        return self.susceptible

    def is_removed(self):
        return self.removed

    def has_transitioned_this_timestep(self):
        return self.transitioned_this_timestep
