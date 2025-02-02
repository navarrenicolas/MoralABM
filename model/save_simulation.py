import numpy as np
import pandas as pd
from MoralABM import MoralABM

# import matplotlib.pyplot as plt

# Import prior data
mf_priors = pd.read_csv('./priors/mf_priors.csv')

# define the moral foundation parameter order
foundations = ['care', 'fairness', 'ingroup', 'authority', 'purity']

# Get all the liberal priors from the response means (Dirichlet)
conservative_dirichlet = mf_priors[mf_priors.bin_pol=='conservative'].groupby('id').apply(
    lambda x : [list(x[x.mf==mf].resp_m)[0]+1 
                for mf in foundations]
)

liberal_dirichlet = mf_priors[mf_priors.bin_pol=='liberal'].groupby('id').apply(
    lambda x : [list(x[x.mf==mf].resp_m)[0]+1 
                for mf in foundations]
)
# irr_priors = mf_priors[mf_priors.bin_pol=='irr'].groupby('id').apply(
#     lambda x : [list(x[x.mf==mf].resp_m)[0] +1
#                 for mf in foundations]
# )

# Get moral foundations priors (Beta)
conservative_betas = mf_priors[mf_priors.bin_pol=='conservative'].groupby('id').apply(
    lambda x : np.array([list(x[x.mf==mf].a) + list(x[x.mf==mf].b)
                for mf in foundations]).flatten()
)

liberal_betas = mf_priors[mf_priors.bin_pol=='liberal'].groupby('id').apply(
    lambda x : np.array([list(x[x.mf==mf].a) + list(x[x.mf==mf].b)
                for mf in foundations]).flatten()
)


beta = True # Use beta distributions for moral foundations
n_sims = 5
n_agent_mean = 100 # mean of normal 
n_agent_scale = 15 # std dev

n_agent_samples = np.random.normal(loc=n_agent_mean, scale=n_agent_scale,size=n_sims)
n_steps = 2001

data_loc = './simulation_data/'

# Set the priors according to the method
conservative_priors = conservative_betas if beta else liberal_dirichlet
liberal_priors = liberal_betas if beta else liberal_dirichlet

for sim in range(n_sims):

    # Create a simulation ID
    sim_id = np.random.randint(100000, 999999)
    
    # gen agent number for this simulation
    n_agents = int(n_agent_samples[sim])
    
    # Run the sim
    abm = MoralABM(n_agents = n_agents,
                   n_steps = n_steps,
                   priors = [conservative_priors,liberal_priors],
                   beta_prior=beta,
                   normalize=True
                  )

    # save the data    
    np.save(data_loc+'mf_'+f'{'beta' if beta else 'dirichlet'}'+'-' + str(sim_id) + '.npy',
            abm.M_agents[:,:,:-1,[n%10==0 for n in range(n_steps)]])

    if beta:
        mf_data = [
            {mf + '_a': abm.M_agents[i,i,2*mfi,t] for mfi,mf in enumerate(foundations)}| {mf + '_b': abm.M_agents[i,i,2*mfi+1,t] for mfi,mf in enumerate(foundations)} | {'id': abm.agent_ids[i]}|{'step':t}|{'bin_pol':['conservative','liberal'][int(i >= n_agents//2)]}
            for i in range(n_agents) for t in range(n_steps) if t%10==0]
    else:
        mf_data = [{mf : abm.M_agents[i,i,mfi,t] for mfi,mf in enumerate(foundations)} | {'id': abm.agent_ids[i]}|{'step':t}|{'bin_pol':['conservative','liberal'][int(i >= n_agents//2)]}
                   for i in range(n_agents) for t in range(n_steps) if t%10==0]
                   
                   
    pd.DataFrame(mf_data).to_csv(data_loc+'moral_abm_'+f'{'beta' if beta else 'dirichlet'}'+'-'+str(sim_id)+'.csv',index=False)
    

    # AMs = np.array([abm.get_adjacency_matrix(step) for step in range(n_steps) if step%10 ==0])
    # Belief_AMs = np.array([abm.get_adjacency_matrix(step,belief=True) for step in range(n_steps) if step%10 ==0])
    AMs = abm.mf_graph[[n%10==0 for n in range(n_steps)],:,:]
    Belief_AMs = abm.belief_graph[[n%10==0 for n in range(n_steps)],:,:]

    
    np.save(data_loc+'mf_graphs_beta-' + str(sim_id) + '.npy', AMs)
    np.save(data_loc+'belief_graphs_beta-' + str(sim_id) + '.npy', Belief_AMs)

