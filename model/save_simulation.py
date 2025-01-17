import numpy as np
import pandas as pd
from MoralABM import MoralABM

# import matplotlib.pyplot as plt

# Import prior data
mf_priors = pd.read_csv('./priors/mf_priors.csv')

# define the moral foundation parameter order
foundations = ['care', 'fairness', 'ingroup', 'authority', 'purity']

# Get all the liberal priors from the response means (Dirichlet)
conservative_priors = mf_priors[mf_priors.bin_pol=='conservative'].groupby('id').apply(
    lambda x : [list(x[x.mf==mf].resp_m)[0]+1 
                for mf in foundations]
)

liberal_priors = mf_priors[mf_priors.bin_pol=='liberal'].groupby('id').apply(
    lambda x : [list(x[x.mf==mf].resp_m)[0]+1 
                for mf in foundations]
)
# irr_priors = mf_priors[mf_priors.bin_pol=='irr'].groupby('id').apply(
#     lambda x : [list(x[x.mf==mf].resp_m)[0] +1
#                 for mf in foundations]
# )

n_sims = 5
n_agent_mean = 100 # mean of normal 
n_agent_scale = 15 # std dev

n_agent_samples = np.random.normal(loc=n_agent_mean, scale=n_agent_scale,size=n_sims)
n_steps = 1000

data_loc = './simulation_data/'

# TODO: Paralelize this
for sim in range(n_sims):

    # Create a simulation ID
    sim_id = np.random.randint(100000, 999999)
    
    # gen agent number for this simulation
    n_agents = int(n_agent_samples[sim])
    
    # Run the sim
    abm = MoralABM(n_agents = n_agents, n_steps = n_steps, priors = [conservative_priors,liberal_priors])
    
    AMs = np.array([abm.get_adjacency_matrix(step) for step in range(n_steps) if step%10 ==0])
    Belief_AMs = np.array([abm.get_adjacency_matrix(step,belief=True) for step in range(n_steps) if step%10 ==0])

    # save the data    
    pd.DataFrame(
        [{mf: abm.M_agents[i,i,mfi,t] for mfi,mf in enumerate(foundations)}|{'id': abm.agent_ids[i]}|{'step':t}|{'bin_pol': ['conservative','liberal'][int(i >= n_agents//2)]} 
         for i in range(n_agents) for t in range(n_steps) if t%10==0]
    ).to_csv(data_loc+'moral_abm-'+str(sim_id)+'.csv',
            index=False)
    
    np.save(data_loc+'mf_graphs-' + str(sim_id) + '.npy', AMs)
    np.save(data_loc+'belief_graphs-' + str(sim_id) + '.npy', Belief_AMs)

