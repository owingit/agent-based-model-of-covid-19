import random
import math


class Policy():
    def __init__(self, hpolicy, mpolicy):
        self.movement_policy = {'policy_name': mpolicy[0],
                                'location_probabilities': mpolicy[1]}
        self.health_policy = hpolicy
        self.policy_distance = 4.0

    def update(self, probabilities_dict):
        self.movement_policy['location_probabilities'] = probabilities_dict

    @property
    def movement_policy_name(self):
        return self.movement_policy.get('policy_name')

    @property
    def movement_probabilities(self):
        return self.movement_policy.get('location_probabilities')

    def get_probability(self, i, location):
        probs = self.movement_probabilities
        probs_at_step = probs.get(i)
        if probs_at_step:
            return probs_at_step[location]
        else:
            raise("No {} probability defined at timestep {}".format(location, i))
