import math
import city
import random
import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

USE_VORONOI = False

class Agent:
    def __init__(self, i, City, beta, gamma):
        '''Defines an agent, which represents a node in the city-level infection network.

        :param int i: num
        :param city.City City: City object encompassing the agent
        :param float beta: experimental beta value
        :param float gamma: experimental gamma denominator
        '''
        # attributes
        self.infection_beta = beta
        self.infection_gamma = gamma

        self.susceptible = True
        self.infected = False
        self.removed = False

        self.timesteps_infected = 0
        self.city = City
        self.health_policy = None
        self.movement_policy = None

        self.positionx = None
        self.positiony = None
        self.prior_x_position = 0
        self.prior_y_position = 0
        self.prior_direction = 0
        self.direction = 0

        self.personal_central_location = [0, 0]
        self.stay_at_home_probability = 6.0 / 7.0  # 6 days out of a week, on average
        self.shop_probability = 1.0 / 7.0
        self.home_location = [0, 0]
        self.movement_state = -1

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
        self.home_location = [self.positionx, self.positiony]

    def move(self):
        '''Move the agent.

        Informed by self.health_policy and self.movement_policy.
        '''
        if self.health_policy == 'social_distancing':
            self.recalculate_vector_based_on_policy()
        if self.movement_policy == '2d_random_walk':
            self.twod_random_walk()

        if self.movement_policy == 'preferential_return':
            self.preferential_return()

        self.recalculate_positions_based_on_edges(self.city)
        self.transitioned_this_timestep = False

    def twod_random_walk(self):
        '''2-d correlated random walk.

        At each timestep an agent chooses a direction - theta - at random and proceeds
        one unit along the vector made by that angle from its current position.
        '''
        self.movement_angle_at_current_timestep = self.theta_star[random.randint(0, 99)]
        self.direction = self.prior_direction + self.movement_angle_at_current_timestep

        #  normal movement, constrained by city boundaries
        self.positionx = self.prior_x_position + (self.velocity * math.cos(self.direction))
        self.positiony = self.prior_y_position + (self.velocity * math.sin(self.direction))

    def preferential_return(self):
        '''Preferential return movement model.

        All agents start at 'home'. With some probability p they choose to leave or stay at home.

        If they leave they teleport to the nearest central location, dictated by the
        voronoi diagram around the poissson point process.

        Once out they are assigned a random number of timesteps (encoded here in movement state) to be out.
        During that time they move with a 2d random walk pattern.

        Once that is done they return home.
        '''
        if self.movement_state == -1:
            if random.random() < self.stay_at_home_probability:
                self.positionx = self.prior_x_position
                self.position_y = self.prior_y_position
            else:
                self.positionx = self.personal_central_location[0]
                self.positiony = self.personal_central_location[1]
                self.movement_state = random.randint(0, 3)
        else:
            if self.movement_state > 0:
                self.twod_random_walk()
                self.movement_state = self.movement_state - 1
            else:
                self.positionx = self.home_location[0]
                self.positiony = self.home_location[1]
                self.movement_state = -1


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

    def reverse_vector(self):
        self.movement_angle_at_current_timestep = random.randint(155, 205) - self.movement_angle_at_current_timestep

    def recalculate_vector_based_on_policy(self):
        self.reverse_vector()  # bounce

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

    def set_central_location(self, points):
        """Set a central location (supermarket) for the agent based on their home location.

        Uses a voronoi diagram where the centers of each region are the points marked by the poisson point process.
        If the home location for an agent is in the voronoi region of a point, that point becomes its central location.
        """
        if USE_VORONOI:
            vor = Voronoi(points)
            vertices = vor.vertices
            regions = vor.regions
            regions.remove([])
            polygons = []
            for i, reg in enumerate(regions):
                polygon_vertices = vertices[reg]
                point_pairs = []
                for pair in polygon_vertices:
                    point_pair = (pair[0], pair[1])
                    point_pairs.append(point_pair)
                polygon = Polygon(point_pairs)
                if i > len(points):
                    i = len(points)
                polygons.append((i, polygon))

            point = Point(self.home_location[0], self.home_location[1])
            assigned = False
            for region, polygon in polygons:
                if polygon.contains(point):
                    self.personal_central_location = list(set([points[region], points[region]]))  # strip dupes
                    assigned = True
            if not assigned:
                self.personal_central_location = list(set([points[(len(regions) / 2)], points[(len(regions) / 2)]]))

            # return to format
            self.personal_central_location = [self.personal_central_location[0][0], self.personal_central_location[0][1]]
        else:
            self.personal_central_location = random.choice(points)

    def set_home_location(self, x, y):
        self.home_location = [x, y]

    def set_policy(self, health_policy, movement_policy):
        self.health_policy = health_policy
        self.movement_policy = movement_policy

    def is_infected(self):
        return self.infected

    def is_susceptible(self):
        return self.susceptible

    def is_removed(self):
        return self.removed

    def has_transitioned_this_timestep(self):
        return self.transitioned_this_timestep
