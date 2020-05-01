from city import *
from CityGraph import *
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys

GAMMAS = np.linspace(1.0, 20.0, num=20)  # infection length (days)
EDGE_PROXIMITIES = np.linspace(0.01, 1.0, num=100)  # proxy for infectivity
DO_PARAMETER_SWEEP = False
COVID_Gamma = 18.0
LOCATION_POLICIES = {
    'lax': {'home': 0.3, 'work': 0.3, 'market': 0.1, 'transit': 0.3},
    'tight': {'home': 0.8, 'work': 0.05, 'market': 0.1, 'transit': 0.05},
    'even': {'home': 0.25, 'work': 0.25, 'market': 0.25, 'transit': 0.25},
    'stay_at_home': {'home': 0.9, 'work': 0.00, 'market': 0.05, 'transit': 0.05},
    'essential_worker': {'home': 0.3, 'work': 0.5, 'market': 0.05, 'transit': 0.15},
    'lockdown': {'home': 0.95, 'work': 0.00, 'market': 0.04, 'transit': 0.01},
}
POLICIES = dict()
migration_prob = (1/100.0)
# migration probability (p) in each city m=p*N will be number of citizens that move from city A to B
MIGRATE = False
SOCIAL_DISTANCING = False
PLOT_SCATTER = False
NRUNS = 5


def main():
    migration_t0 = 1
    lockdown_t0 = 15
    lockdown_t0s = [lockdown_t0]
    '''
    for t in T:
        datafile='data/imax{}.dat'.format(t)
        fn=open(datafile,'w+')
        fn.write('\n')
        fn.close()
    '''
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for t in lockdown_t0s:
        lockdown_t0 = t
        if DO_PARAMETER_SWEEP:
            for edge_proximity in EDGE_PROXIMITIES:
                for gamma in GAMMAS:
                    setup_and_run(timesteps, edge_proximity, gamma, migration_t0, lockdown_t0)
        else:
            nruns = NRUNS
            #imax = np.zeros((2,nruns))                                #ADD number of cities to first index of array
            datafile = 'data/imax{}.dat'.format(t)
            fn = open(datafile, 'a+')
            imaxs = []
            for i in range(nruns):  
                i_max = setup_and_run(timesteps, 0.2, COVID_Gamma, migration_t0, lockdown_t0)
                #  print(Imax)
                imaxs.append(i_max[0])
                #  imaxs[:][i]=i_max[:]
                #  imaxs[0][i]=i_max[0]
                #  imaxs[1][i]=i_max[1]
                fn.write("{}\t{}\n".format(i, i_max[0]))
            fn.close()
            print(imaxs)
    #print(np.mean(imaxs[0][:]), np.mean(imaxs[1][:]))


def setup_and_run(timesteps, edge_proximity, gamma):
    """Initialize the simulation.

    :param timesteps: number of timesteps to run the simulation
    :param edge_proximity: edge proximity parameter (proxy for infection rate)
    :param gamma: gamma parameter (proxy for recovery rate)
    :param migration_threshold: timestep in which migration occurs, if enabled
    :param lockdown_threshold: timestep at which lockdown occurs, if enabled
    """
    cities = construct_cities(edge_proximity, gamma, timesteps, lockdown_threshold)

    for city_i in cities:
        city_i.set_initial_states()
    city_graphs = []
    for city_i in cities:
        city_graphs.append(CityGraph(city_i))
    for i in range(0, timesteps):
        for city_i, city_graph in zip(cities, city_graphs):
            if SOCIAL_DISTANCING:
                if i > lockdown_threshold:
                    city_i.change_proximity(edge_proximity*0.5)
            city_graph.xs.append(i)
            print('Day {} - {}'.format(i, city_i.name))
            beta = city_i.timestep(i) / city_i.N
            if MIGRATE:
                if i < migration_threshold:
                    migration(cities)

            city_graph.set_beta(beta)
            state_dict = city_i.get_states()
            if state_dict['total_IR'] == city_i.N:
                if not city_graph.timestep_of_convergence:
                    city_graph.timestep_of_convergence = i

            city_graph.ys.append(state_dict)
            city_i.print_states()
            if PLOT_SCATTER:
                city_i.plot_scatter(i)
        finished_cities = []
        for city in cities:
            if city.num_infected == 0:
                finished_cities.append(city)
        if len(finished_cities) == len(cities):
            print('All agents are free of infection.')
            break

    i_max = []

    for cg in city_graphs:
        cg.total_infected = cg.ys[len(cg.xs)-1]['total_IR']
        infected = cg.plot_data()
        i_max.append(infected)
        cg.plot_ro()
    print(i_max)
    return i_max


def construct_cities(edge_proximity, gamma_denom, timesteps, lockdown_threshold):
    """Initialize cities with different policies, beta, gamma values. Right now they're all the same size/population

    :param float edge_proximity: experimental edge_proximity value
    :param float gamma_denom: gamma denominator
    :param int timesteps
    :param int lockdown_threshold: num timesteps at which to initiate lockdown
    :returns: list of city objects"""

    # TODO: different Ro for different cities, based on data?
    gamma = 1.0 / gamma_denom

    intent = 'tight'
    location_policies_dict_a = construct_location_policies_dict(intent, timesteps, lockdown_threshold)
    mpolicy_a = ['preferential_return_tight', location_policies_dict_a]
    intent = 'stay_at_home'
    location_policies_dict_b = construct_location_policies_dict(intent, timesteps, lockdown_threshold)
    mpolicy_b = ['preferential_return_stay_at_home', location_policies_dict_b]
    intent = 'lockdown'
    location_policies_dict_c = construct_location_policies_dict(intent, timesteps, lockdown_threshold)
    mpolicy_c = ['preferential_return_lockdown', location_policies_dict_c]
    intent = 'lax'
    location_policies_dict_d = construct_location_policies_dict(intent, timesteps, lockdown_threshold)
    mpolicy_d = ['preferential_return', location_policies_dict_d]
    intent = 'restrict'
    location_policies_dict_e = construct_location_policies_dict(intent, timesteps, lockdown_threshold)
    mpolicy_e = ['preferential_return', location_policies_dict_e]
    intent = 'even'
    location_policies_dict_f = construct_location_policies_dict(intent, timesteps, lockdown_threshold)
    mpolicy_f = ['preferential_return_even', location_policies_dict_f]
    
    frequencies_dict_a = {'market': 50, 'transit': 200, 'work': 20, 'home': 2}

    ws = [200, 300, 50]
    hs = [200, 300, 50]
    ns = [1500, 5000, 600]
    hpolicy_a = 'social_distancing'
    hpolicy_b = 'normal'
    cities = [
              # City('Boulder', ws[0], hs[0], ns[1], edge_proximity, gamma, hpolicy_b, mpolicy_d,
              #     frequencies_dict_a),
              # City('Denver', ws[1], hs[1], ns[1], edge_proximity, gamma, hpolicy_b, mpolicy_b,
              #     frequencies_dict_a),
              # City(name='Quarantinopolis', x=ws[1], y=hs[1], n=ns[1], edge_proximity=edge_proximity,
              #      gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_c),
              # City(name='EssentialWorkerOpolis', x=ws[1], y=hs[1], n=ns[1], edge_proximity=edge_proximity,
              #      gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_d),
              City('City A', ws[0], hs[0], ns[0], edge_proximity, gamma, hpolicy_b, mpolicy_e,
                   frequencies_dict_b)]
    for city_i in cities:
        city_i.view_all_policies(POLICIES)
    return cities


def construct_location_policies_dict(intent, timesteps):
    """Make policy different at each timestep."""
    location_policies_dict = {}
    for i in range(0, timesteps):
        if i < 10:
             location_policies_dict[i] = LOCATION_POLICIES['lax']
        else:
            location_policies_dict[i] = LOCATION_POLICIES[intent]
    POLICIES[intent] = location_policies_dict

    return location_policies_dict


def migration(cities):
    """
    TODO: Figure out migration for preferential movement and get rid of this brute forcing
    """
    for city_pair in list(itertools.combinations(cities, r=2)):
        migrants=((int)(migration_prob*max(city_pair[0].N,city_pair[0].N)))
     #   print(migrants)
        for m in range(migrants): 
            mgrnt0=random.choice(city_pair[0].agents)
            mgrnt1=random.choice(city_pair[1].agents)
            shuffle(city_pair[0],city_pair[1],mgrnt0,mgrnt1)
        
            #Source to target city
    #    print(city_pair[0].name,city_pair[1].name)


def shuffle(source_city, target_city, agent0, agent1):
    agent1_prior_state = agent1.state
    # print(st)
    migrating_agents = [agent0, agent1]
    for m in migrating_agents:
        cty = m.get_city()
        if m.state == "susceptible":
            cty.num_susceptible -= 1
        if m.state == "infected":
            cty.num_infected -= 1
        if m.state == "removed":
            cty.num_removed -= 1
    agent1.transition_state(agent0.state)
    agent0.transition_state(agent1_prior_state)
    #m0.positionx = random.random()*((source.width) *0.2 )
    migrating_agents_modified = [agent0, agent1]
    shuffle_central_locations(agent0, agent1)
    for m in migrating_agents_modified:
        cty = m.get_city()
        '''
        Sending migrant individuals to around their home location
        '''
        m.positionx = list(m.personal_central_locations['home'])[0] + np.random.normal(-0.5, 0.5)
        m.positiony = list(m.personal_central_locations['home'])[1] + np.random.normal(-0.5, 0.5)

        if m.state == "susceptible":
            cty.num_susceptible += 1
        if m.state == "infected":
            cty.num_infected += 1
        if m.state == "removed":
            cty.num_removed += 1


def shuffle_central_locations(agent0, agent1):
    '''
    Brute Forced through the shuffling
    TODO: Make the shuffling code more elegant
    '''
    migrating_agents = [agent0, agent1]
    modes = ['market', 'home', 'work', 'transit']
    for mode in modes:
        m0x = list(agent0.personal_central_locations[mode])[0]
        m0y = list(agent0.personal_central_locations[mode])[1]
        m1x = list(agent1.personal_central_locations[mode])[0]
        m1y = list(agent1.personal_central_locations[mode])[1]
        agent0.personal_central_locations[mode] = [m1x, m1y]
        agent1.personal_central_locations[mode] = [m0x, m0y]


if __name__ == "__main__":
    main()
