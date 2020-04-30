# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:25:20 2020

@author: atiyab
"""


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
    'restrict' : {'home': 0.75, 'work': 0.05, 'market': 0.1, 'transit': 0.1}
}
POLICIES = dict()
migration_prob = (1/100.0) 

'''
migration probability (p) in each city 
m=p*N will be number of citizens that move from city A to B
for simplicity I assume for now that m people move from A to B 
and same amount moves from B to A.
TODO: Make it  more realistic with source and target specified 
and different number of people travelling in different direction
'''

def main():
    migration_t0=1
    lockdown_t0=25
    T=[25]
    '''
    for t in T:
        datafile='data/imax{}.dat'.format(t)
        fn=open(datafile,'w+')
        fn.write('\n')
        fn.close()
    '''
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for t in T:
        lockdown_t0=t
        if DO_PARAMETER_SWEEP:
            for edge_proximity in EDGE_PROXIMITIES:
                for gamma in GAMMAS:
                    setup_and_run(timesteps, edge_proximity, gamma) 
        else:
            nruns=5
            #imax=np.zeros((2,nruns))                                #ADD number of cities to first index of array
            datafile='data/imax{}.dat'.format(t)
            fn=open(datafile,'a+')
            imax=[]
            for i in range(nruns):  
                Imax=setup_and_run(timesteps, 0.2, COVID_Gamma,migration_t0,lockdown_t0)
                #print(Imax)
                imax.append(Imax[0])
     #           imax[:][i]=Imax[:]
                #imax[0][i]=Imax[0]
               # imax[1][i]=Imax[1]
                fn.write("{}\t{}\n".format(i,Imax[0]))
            fn.close()
            print(imax)
    #print(np.mean(imax[0][:]),np.mean(imax[1][:]))
        

def setup_and_run(timesteps, edge_proximity, gamma , migration_threshold,lockdown_threshold):
    """Initialize the simulation.

    :param timesteps: number of timesteps to run the simulation
    :param edge_proximity: edge proximity parameter (proxy for infection rate)
    :param gamma: gamma parameter (proxy for recovery rate)
    """
    cities = construct_cities(edge_proximity, gamma, timesteps,lockdown_threshold)
    for city in cities:
        city.set_initial_states()
    city_graphs = []
    for city in cities:
        city_graphs.append(CityGraph(city))
    t0=migration_threshold
    for i in range(0, timesteps):
        for city, city_graph in zip(cities, city_graphs):
            ''' Uncomment the following for triggering social Distancing '''
            #if i>lockdown_threshold:
            #    city.change_proximity(edge_proximity*0.5)
            city_graph.xs.append(i)
            print('Day {} - {}'.format(i, city.name))
            beta = city.timestep(i) / city.N
           # if i<t0:    
            #    migration(cities)            
            city_graph.set_beta(beta)
            state_dict = city.get_states()
            if state_dict['total_IR'] == city.N:
                if not city_graph.timestep_of_convergence:
                    city_graph.timestep_of_convergence = i

            city_graph.ys.append(state_dict)
            city.print_states()
            #city.plot_scatter(i)
        finished_cities = []
        for city in cities:
            if city.num_infected == 0:
                finished_cities.append(city)
        if len(finished_cities) == len(cities):
            print('All agents are free of infection.')
            break
    Imax=[]
    for cg in city_graphs:
        cg.total_infected = cg.ys[len(cg.xs)-1]['total_IR']
        I=cg.plot_data()
        Imax.append(I)
        cg.plot_ro()
    print(Imax)
    return Imax
        #cg.write_data()


def construct_cities(edge_proximity, gamma_denom, timesteps,t0):
    """Initialize cities with different policies, beta, gamma values. Right now they're all the same size/population

    :param float edge_proximity: experimental edge_proximity value
    :param float gamma_denom: gamma denominator
    :param int timesteps
    :returns: list of city objects"""
    # TODO: different Ro for different cities, based on data?
    gamma = 1.0 / gamma_denom
    
    intent = 'tight'
    location_policies_dict_a = construct_location_policies_dict(intent, timesteps,t0)
    mpolicy_a = ['preferential_return_tight', location_policies_dict_a]
    intent = 'stay_at_home'
    location_policies_dict_b = construct_location_policies_dict(intent, timesteps,t0)
    mpolicy_b = ['preferential_return_stay_at_home', location_policies_dict_b]
    intent = 'lockdown'
    location_policies_dict_c = construct_location_policies_dict(intent, timesteps,t0)
    mpolicy_c = ['preferential_return_lockdown', location_policies_dict_c]
    intent = 'lax'
    location_policies_dict_d = construct_location_policies_dict(intent, timesteps,t0)
    mpolicy_d = ['preferential_return', location_policies_dict_d]
    intent = 'restrict'
    location_policies_dict_e = construct_location_policies_dict(intent, timesteps,t0)
    mpolicy_e = ['preferential_return', location_policies_dict_e]
    intent = 'even'
    location_policies_dict_f = construct_location_policies_dict(intent, timesteps,t0)
    mpolicy_f = ['preferential_return_even', location_policies_dict_f]
    
    frequencies_dict_a = {'market': 50, 'transit': 200, 'work': 20, 'home': 2}
    frequencies_dict_b = {'market': 50, 'transit': 200, 'work': 20, 'home': 3}
    '''
    New Zeeland 18/km*2
    here 0.2  edge proximity equal to 2 m in real life: 1 km = 100 units
    area=200*200 = 4e4 units^2 = 400 km^2 population =18*400
    '''
    ws = [200, 300, 50]
    hs = [200, 300, 50]
    #ns = [1500, 500, 600]
    ns = [600, 500, 600]
    hpolicy_a = 'social_distancing'
    hpolicy_b = 'normal'
    cities = [#City('Boulder', ws[0], hs[0], ns[1], edge_proximity, gamma, hpolicy_b, mpolicy_d,
    #               frequencies_dict_a),
    #           City('Denver', ws[1], hs[1], ns[1], edge_proximity, gamma, hpolicy_b, mpolicy_b,
    #                frequencies_dict_a),
    #           City(name='Quarantinopolis', x=ws[1], y=hs[1], n=ns[1], edge_proximity=edge_proximity,
    #                gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_c),
    #           City(name='EssentialWorkerOpolis', x=ws[1], y=hs[1], n=ns[1], edge_proximity=edge_proximity,
    #                gamma=gamma, hpolicy=hpolicy_b, mpolicy=mpolicy_d),
              City('City A', ws[0], hs[0], ns[0], edge_proximity, gamma, hpolicy_b, mpolicy_e,
                   frequencies_dict_b)]
    #          City('Colorado Springs', ws[1], hs[1], ns[0], edge_proximity, gamma, hpolicy_b, mpolicy_a,
    #               frequencies_dict_a)]
              # City('DenserBoulder', ws[0], hs[0], ns[1], beta, gamma, hpolicy_a, mpolicy_a),
              # City('DenserDenver', ws[1], hs[1], ns[1], beta, gamma, hpolicy_b, mpolicy_b),
              # City('DenserFort Collins', ws[0], hs[0], ns[1], beta, gamma, hpolicy_a, mpolicy_b),
              # City('DenserColorado Springs', ws[1], hs[1], ns[1], beta, gamma, hpolicy_b, mpolicy_a)]
    for city in cities:
        city.view_all_policies(POLICIES)
    return cities


def construct_location_policies_dict(intent, timesteps,t0):
    """Make policy different at each timestep."""
    location_policies_dict = {}
    for i in range(0, timesteps):
        if i < t0:
            location_policies_dict[i] = LOCATION_POLICIES['lax']
       # elif (i < t0+2 and i>t0):
      #      location_policies_dict[i] = LOCATION_POLICIES['restrict']
        else:
            location_policies_dict[i] = LOCATION_POLICIES[intent]
    POLICIES[intent] = location_policies_dict
    return location_policies_dict
'''
TODO: Figure out migration for preferential movement and get rid of this brute forcing

'''
def migration(cities):
    for city_pair in list(itertools.combinations(cities, r=2)):
        migrants=((int)(migration_prob*max(city_pair[0].N,city_pair[0].N)))
     #   print(migrants)
        for m in range(migrants): 
            mgrnt0=random.choice(city_pair[0].agents)
            mgrnt1=random.choice(city_pair[1].agents)
            shuffle(city_pair[0],city_pair[1],mgrnt0,mgrnt1)
        
            #Source to target city
    #    print(city_pair[0].name,city_pair[1].name)
            
def shuffle(source,target,m0,m1):
    st=m1.state
   # print(st)
    M=[m0,m1]
    for m in M:
        cty=m.get_city
      #  print(cty.name)
        if m.state == "susceptible":
            cty.num_susceptible -=1
        if m.state == "infected":
            cty.num_infected -=1
        if m.state == "removed":
            cty.num_removed -=1
    m1.transition_state(m0.state)
    m0.transition_state(st)
    #m0.positionx = random.random()*((source.width) *0.2 )
    M=[m0,m1]
    shuffle_central_locations(m0,m1)
    for m in M:
        cty=m.get_city
        '''
        Sending migrant individuals to around their home location
        '''
        m.positionx=list(m.personal_central_locations['home'])[0] + np.random.normal(-0.5, 0.5)         
        m.positiony=list(m.personal_central_locations['home'])[1] + np.random.normal(-0.5, 0.5)

        if m.state == "susceptible":
            cty.num_susceptible +=1
        if m.state == "infected":
            cty.num_infected +=1
        if m.state == "removed":
            cty.num_removed +=1   
            
'''
Brute Forced through the shuffling
TODO: Make the shuffling code more elegant
'''
def shuffle_central_locations(m0,m1):
    M=[m0,m1]
    modes=['market','home','work','transit']
    for mode in modes:
        m0x=list(m0.personal_central_locations[mode])[0]
        m0y=list(m0.personal_central_locations[mode])[1]
        m1x=list(m1.personal_central_locations[mode])[0]
        m1y=list(m1.personal_central_locations[mode])[1]
        m0.personal_central_locations[mode]=[m1x,m1y]
        m1.personal_central_locations[mode]=[m0x,m0y]
        
if __name__ == "__main__":
    main()
