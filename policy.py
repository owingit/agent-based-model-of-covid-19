import random
import math


class Policy():
    def __init__(self, name, agents, hpolicy, mpolicy, area):
        self.city = name
        self.agents = agents
        self.movement_policy = mpolicy
        self.health_policy = hpolicy
        self.policy_distance = 4.0

    def unset_health_policy(self):
        self.health_policy = None
