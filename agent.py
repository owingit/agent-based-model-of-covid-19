import math
import city
import random
import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Agent:
    def __init__(self, i, City):
        '''Defines an agent, which represents a node in the city-level infection network.

        :param int i: num
        :param city.City City: City object encompassing the agent
        :param **kwargs
        '''
        # attributes
        self.susceptible = True
        self.infected = False
        self.removed = False

        self.timesteps_infected = 0
        self.city = City
        self.health_policy = None
        self.movement_policy = None
        self.health_policy_active = False

        self.positionx = None
        self.positiony = None
        self.prior_x_position = 0
        self.prior_y_position = 0
        self.prior_direction = 0
        self.direction = 0

        self.mode = None
        self.personal_central_locations = {}
        self.stay_at_home_probability = None # 6 days out of a week, on average
        self.work_probability = None
        self.transit_probability = None
        self.shop_probability = None

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

    def move(self):
        '''Move the agent.

        Informed by self.health_policy and self.movement_policy.
        '''
        if self.health_policy == 'social_distancing' and self.health_policy_active:
            self.recalculate_vector_based_on_policy()
        if self.movement_policy[0] == '2d_random_walk':
            self.twod_random_walk()

        if self.movement_policy[0] == 'preferential_return':
            self.preferential_return()

        self.recalculate_positions_based_on_edges(self.city)
        self._initialize_dynamic_state(False)
        self.deactivate_health_policy()

    def _initialize_dynamic_state(self, state):
        """Measures whether an infection state transition has happened during this timestep for this agent."""
        self.transitioned_this_timestep = state

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
        '''
        msg = '{} probability has not been set!'
        assert self.shop_probability, msg.format('market')
        assert self.stay_at_home_probability, msg.format('home')
        assert self.work_probability, msg.format('work')
        rand_val = random.random()
        if rand_val < self.stay_at_home_probability:
            mode = 'home'
        elif self.stay_at_home_probability + self.work_probability > rand_val >= self.stay_at_home_probability:
            mode = 'work'
        elif self.shop_probability + self.stay_at_home_probability + self.work_probability > rand_val >= self.stay_at_home_probability + self.work_probability:
            mode = 'market'
        else:
            mode = 'transit'
        self.mode = mode
        # print('{} in {} mode in {} state'.format(self.name, self.mode, self.state))
        assert self.mode
        self.prior_x_position = self.positionx
        self.prior_y_position = self.positiony
        self.positionx = list(self.personal_central_locations[self.mode])[0]
        self.positiony = list(self.personal_central_locations[self.mode])[1]

    def recalculate_positions_based_on_edges(self, city):
        '''Adjust the positions of an agent based on the city's boundaries.

        :param city.City city: city with edges against which to compare location for reflection
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

        if x_modified and y_modified:
            self.reverse_vector()

    def recalculate_vector_based_on_policy(self):
        self.reverse_vector()  # bounce

    def reverse_vector(self):
        self.movement_angle_at_current_timestep = random.randint(155, 205) - self.movement_angle_at_current_timestep

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

    def set_and_verify_locations(self, market_regions, transit_regions, workspace_regions, home_regions):
        """Set a central location (supermarket) for the agent based on their home location.


        If the home location for an agent is in the voronoi region of a point, that point becomes its central location.

        :param tuple market_regions: list of points denoting 'market' central locations and their regional boundaries
        :param tuple transit_regions: list of points denoting 'transit_hub' central locations and their regional boundaries
        :param tuple workspace_regions: list of points denoting 'work' central locations and their regional boundaries
        :param tuple home_regions: list of points denoting 'home' central locations and their regional boundaries
        """
        point = Point(self.positionx, self.positiony)

        enumerated_regions = {'market': market_regions[0],
                              'transit': transit_regions[0],
                              'work': workspace_regions[0],
                              'home': home_regions[0]
                              }

        enumerated_points = {'market': market_regions[1],
                             'transit': transit_regions[1],
                             'work': workspace_regions[1],
                             'home': home_regions[1]}
        # update the agent's personal central location for each mode
        used_regions = {'market': None,
                        'transit': None,
                        'work': None,
                        'home': None
                        }

        for id, poly_tuples in enumerated_regions.items():
            assigned = False
            for region, polygon in poly_tuples:
                if polygon.contains(point) and not assigned:
                    self.personal_central_locations[id] = frozenset([enumerated_points[id][region][0], enumerated_points[id][region][1]])  # strip dupes
                    assigned = True
                    used_regions[id] = region

            if not assigned:
                random_index = random.randint(0, len(poly_tuples)) % len(enumerated_points[id])
                self.personal_central_locations[id] = frozenset([enumerated_points[id][random_index][0], enumerated_points[id][random_index][1]])
                used_regions[id] = random_index

        return used_regions

    def set_policy(self, health_policy, movement_policy, i):
        self.health_policy = health_policy
        self.movement_policy = movement_policy
        if self.movement_policy[1]:
            self.shop_probability = self.movement_policy[1][i]['market']
            self.stay_at_home_probability = self.movement_policy[1][i]['home']
            self.transit_probability = self.movement_policy[1][i]['transit']
            self.work_probability = self.movement_policy[1][i]['work']

    def is_infected(self):
        return self.infected

    def activate_health_policy(self):
        self.health_policy_active = True

    def deactivate_health_policy(self):
        self.health_policy_active = False

    def is_susceptible(self):
        return self.susceptible

    def is_removed(self):
        return self.removed

    def has_transitioned_this_timestep(self):
        return self.transitioned_this_timestep
