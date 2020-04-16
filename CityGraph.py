import json


class CityGraph:
    def __init__(self, city):
        self.name = city.name
        self.beta = city.beta
        self.gamma = city.gamma
        self.N = city.N
        self.datafile = city.datafile
        self.xs = []
        self.ys = []
        self.timestep_of_convergence = None
        self.total_infected = 0

    def write_data(self):
        data_line = '{} {}\n'.format(self.beta * (1 / self.gamma), self.total_infected)
        with open(self.datafile, 'a+',) as f:
            f.write(data_line)
