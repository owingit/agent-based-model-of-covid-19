# ABM for COVID-19

# Topic: 
Quantification of the effect of various public health measures on the disease dynamics through agent-based modeling. Agent-based modeling for COVID-19

# To get started:
1. ```pip install -r requirements.txt```
2. Edit the 'simulation.py' file to parameterize the cities as you see fit. The values currently in there resolve in a reasonable amount of simulation time.
3. From the command line, run ```python simulation.py```
4. Some of the things that you can toggle right now:
   a) Change the policy to and from social distancing
   b) Change the movement policy to and from 2d random walk / preferential return
5. Each city will plot its SIR curve / time at the end of the simulation, in order of creation.
