import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize

import networkx as nx
from sklearn.cluster import SpectralClustering


sim_type = 'beta'

def get_filenames(directory):

    try:
        return os.listdir(directory)
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
        return []


directory_path = f'./data/example_data/simulation_data/'  
filenames = get_filenames(directory_path)

def get_sim_number(filename):
    result = ""
    for char in filename:
        if char.isdigit():      
            result += char
    return result

#####################
## Normalise graphs
#####################

mf_graphs = [file for file in filenames if 'mf_graph' in file]
belief_graphs = [file for file in filenames if 'belief_graph' in file]

all_sims = [get_sim_number(file) for file in belief_graphs]
sim_nums = all_sims

def logistic(x,b=1):
    return 1/(1+np.exp(-b*x))
    
def normalize_network(graph,b=1):
    return np.exp(-b*graph)

def normalization_variance(param,graph):
    b = np.exp(-param)
    return -np.sum([normalize_network(graph,b=b)[i].var() for i in range(graph.shape[0])])

def fit_normalizing_param(graph):
    fit  = minimize(normalization_variance,
               [0.5], # start_point
               args = (graph),
               bounds = [(None,None)],
              )
    return np.exp(-fit.x),fit

## Improvised graph loader from the simulation number
def load_sim(sim_id,normalize=False,belief=True):
    
    graph_name = 'belief' if belief else 'mf'
    
    graph = np.load(directory_path + graph_name + '_graphs_beta-' + str(sim_id)+ '.npy')
    
    if graph.shape[0] == graph.shape[1]:
        graph = np.rot90(graph,k=1,axes=(0,2))
        
    
    if normalize:
        query = 'belief_norm' if belief else 'mf_norm'
        if sim_id in norming_df.sim_id:
            b = norming_df[norming_df.sim_id==sim_id][query].iloc[0]
        else:
            b = norming_df.mean(axis=0)[query]
        graph = normalize_network(graph,b=b)
                
    return graph

## Start Normalization

def get_norm_fits(sim_numbers):
    norm_fits = []
    for i,sim_id in enumerate(sim_numbers): # only 50 simulations
        try: 
            d = {'belief_norm': fit_normalizing_param(load_sim(sim_id,normalize=False,belief=True))[0],
             'mf_norm': fit_normalizing_param(load_sim(sim_id,normalize=False,belief=False))[0],
             'sim_id':  sim_id,
            } 
        except: 
            print(f'Failed to fit param for {i}')
            continue
        norm_fits.append(d)
    return norm_fits




norm_fits = get_norm_fits(sim_nums)
norming_df = pd.DataFrame(norm_fits)
norming_df = norming_df.assign(
    belief_norm = norming_df.belief_norm.apply(lambda x: x[0]),
    mf_norm = norming_df.mf_norm.apply(lambda x: x[0])
)
os.makedirs( f'{directory_path}../graph_data', exist_ok=True)
norming_df.to_csv(f'{directory_path}../graph_data/graph_norming_data-beta.csv',index=False)

#####################
# Get minimum clusters
#####################


data_loc = f'{directory_path}../graph_data/'
norming_file = f'graph_norming_data-{sim_type}.csv'
norming_df = pd.read_csv(data_loc+norming_file)


def simplify_graph(AM,method='max',diag_fill = 0):
    outgoing = np.triu(AM)+np.tril(AM.T)
    incoming = np.tril(AM)+np.triu(AM.T)
    
    if method =='out':
        new_AM = outgoing
    elif method =='in':
        new_AM = incoming
    elif method == 'max':
        new_AM = np.maximum(outgoing, incoming)
    else:
        new_AM = np.minimum(outgoing, incoming)
    
    np.fill_diagonal(new_AM,diag_fill)
    return new_AM

def laplacian(AM):
    D = np.diag(AM.sum(axis=1))
    return D - AM

def laplacian_eigen(AM):
    L = laplacian(
        simplify_graph(AM,diag_fill=0)
    )
    eigens_sorted = np.sort(np.linalg.eigvals(L))
    return eigens_sorted

def cluster_number(sim_id,belief=True):
    
    AMs = load_sim(sim_id,normalize=True,belief=belief) # similarity
    return [
        np.argmax(
            np.diff(
                laplacian_eigen(AMs[i])[1:]
            )[1:]
        )+2 for i in range(AMs.shape[0])]

def get_min_clusters(sims,belief=True):
    min_clusters = []
    for sim_id in sims:
        min_clusters.append(
            {'sim_id' : sim_id,
             'min_clusters': cluster_number(sim_id,belief)}
        )
    return min_clusters


if os.path.isfile(data_loc + f'minimum_belief_clusters-{sim_type}.csv'):
    print('loading min clusters')
    min_clusters_df = pd.read_csv(data_loc + f'minimum_belief_clusters-{sim_type}.csv')
    min_clusters_df = min_clusters_df.assign(
        min_clusters=min_clusters_df.min_clusters.apply(json.loads)
    )
else:
    print('computing min clusters')
    min_clusters = get_min_clusters(sim_nums,belief=False)
    min_clusters_df = pd.DataFrame(min_clusters)
    min_clusters_df = min_clusters_df.assign(
        min_clusters=min_clusters_df.min_clusters.apply(str)
    )
    min_clusters_df.to_csv(data_loc + f'minimum_belief_clusters-{sim_type}.csv',index=False)


if type(min_clusters_df.min_clusters.iloc[0])==str:
    min_clusters_df = min_clusters_df.assign(
        min_clusters = min_clusters_df.min_clusters.apply(json.loads)
    )
if type(min_clusters_df.sim_id.iloc[0])!=str:
    min_clusters_df = min_clusters_df.assign(
        sim_id = min_clusters_df.sim_id.apply(str)
    )
 

def get_clusters(sims,min_clusters_df):
    clusters = []
    for sim_id in sims:
        print('new file: ', sim_id)
        AMs = load_sim(sim_id,normalize=True,belief=True)

        for t in range(AMs.shape[0]):
            
            sym_graph = simplify_graph(AMs[t],method='min')
            n_clusters = min_clusters_df[min_clusters_df.sim_id==(sim_id)].min_clusters.iloc[0][t]
            clusters.append(
                {'sim_id': sim_id, 'step': 10*t,
                  'clusters': SpectralClustering(n_clusters = n_clusters,
                                                 n_components= n_clusters,
                                                 affinity='precomputed',
                                                 assign_labels='kmeans').fit_predict(sym_graph)
                 }
            )
        
    return clusters

belief_clusters = get_clusters(sim_nums,min_clusters_df)
belief_clusters_df = pd.DataFrame(belief_clusters)
belief_clusters_df = belief_clusters_df.assign(n_clusters = belief_clusters_df.clusters.apply(lambda x: np.unique(x).shape[0]))
belief_clusters_df.assign(
    clusters = belief_clusters_df.clusters.apply(lambda x: str(list(x)))
).to_csv(data_loc+f'belief_clusters-{sim_type}.csv',index=False)

# Analyzing cluster data

# norming_df = pd.read_csv(f'{directory_path}../graph_data/graph_norming_data-{sim_type}.csv')


def compute_homogeity(cluster_df):

    ingroup_homogeneity = []

    # for names, group in cluster_df.groupby(['sim_id','step']):
    for sim_id,sim_group in cluster_df.groupby('sim_id'):

        for step,step_group in sim_group.groupby('step'):
            # Convert to numerical list incase the saved dataset turned the cluster
            # into a string
            clusters = step_group.clusters.iloc[0]

            homogeneities = np.zeros(len(clusters))
            sim_size = len(clusters)

            for agent_i,agent in enumerate(clusters):

                cluster = np.argwhere(np.array(clusters)==agent).T

                cluster_size = cluster.shape[1]

                agent_bin_pol  = agent_i >= sim_size//2 # 0 cons, 1 lib

                cluster_bin_pol = cluster >= sim_size//2

                same_bin_pols = cluster_bin_pol == agent_bin_pol

                homogeneities[agent_i] = (np.sum(same_bin_pols)-1)/cluster_size

            ingroup_homogeneity.append(
                {'step': step, 'sim_id': sim_id, 'homogeneity': list(homogeneities)}
            )


    return ingroup_homogeneity


def compute_ingroup_accuracy(cluster_df):

    ingroup_accs = []

    # for names, group in cluster_df.groupby(['sim_id','step']):
    for sim_id,sim_group in cluster_df.groupby('sim_id'):
        actual_graph = load_sim(sim_id,normalize=True,belief=False)
        for step,step_group in sim_group.groupby('step'):

            actual_simple = simplify_graph(actual_graph[int(step/10)],method='max',diag_fill=1)

            # Convert to numerical list incase the saved dataset turned the cluster
            # into a string
            clusters = step_group.clusters.iloc[0]

            accuracies = np.zeros(len(clusters))
            for agent_i,agent in enumerate(clusters):

                cluster = np.argwhere(np.array(clusters)==agent).T
                cluster_size = cluster.shape[1]
                accuracies[agent_i] = actual_simple[agent_i,cluster].mean()*cluster_size/len(clusters)

            ingroup_accs.append(
                {'step': step, 'sim_id': sim_id, 'accuracies': list(accuracies)}
            )


    return ingroup_accs


## Compute data

def get_homog_df(df):
    homog = compute_homogeity(df)

    homog_df = pd.DataFrame(homog)

    homog_df = homog_df.assign(
        homog_cons = homog_df.homogeneity.apply(lambda x: np.mean(x[:len(x)//2])),
        homog_lib = homog_df.homogeneity.apply(lambda x: np.mean(x[len(x)//2:])),
    )
    return homog_df

def get_acc_df(df):
    test = compute_ingroup_accuracy(df)
    test_df = pd.DataFrame(test)

    acc_df = test_df.assign(
        acc_cons = test_df.accuracies.apply(lambda x: np.mean(x[:len(x)//2])),
        acc_lib = test_df.accuracies.apply(lambda x: np.mean(x[len(x)//2:])),
    )
    return acc_df


def fix_cluster_df(df):
    if type(df.clusters.iloc[0])==str:
        df = df.assign(
            clusters = df.clusters.apply(json.loads)
        )
    return df

belief_clusters = pd.read_csv(f'{directory_path}../graph_data/belief_clusters-beta.csv')
belief_clusters = fix_cluster_df(belief_clusters)
n_steps = belief_clusters.value_counts('step').shape[0]


acc_df  = get_acc_df(belief_clusters)
homog_df  = get_homog_df(belief_clusters)

def melt_acc_homog(acc_df,homog_df):
    # 1. Melt the DataFrames to combine 'cons' and 'lib' columns
    acc_melted = acc_df.melt(id_vars=['step', 'sim_id'], value_vars=['acc_cons', 'acc_lib'], var_name='bin_pol', value_name='acc_value')
    homog_melted = homog_df.melt(id_vars=['step', 'sim_id'], value_vars=['homog_cons', 'homog_lib'], var_name='bin_pol', value_name='homog_value')
    
    # 2. Extract the 'lib' or 'cons' label from the 'type' column
    acc_melted['bin_pol'] = acc_melted['bin_pol'].str.replace('acc_', '')  # Remove 'acc_' prefix
    homog_melted['bin_pol'] = homog_melted['bin_pol'].str.replace('homog_', '') # Remove 'homog_' prefix
    
    # 3. Merge the two melted DataFrames based on 'step', 'sim_id', and 'type'
    merged_df = pd.merge(acc_melted, homog_melted, on=['step', 'sim_id', 'bin_pol'], how='inner')
    
    # 4. Rename columns for clarity (optional)
    merged_df = merged_df.rename(columns={'acc_value': 'acc', 'homog_value': 'homog'})
    
    # Final Result: merged_df
    return merged_df

acc_homog_df = melt_acc_homog(acc_df,homog_df)
plot_data_loc = f'{directory_path}../plot_data/'
os.makedirs(plot_data_loc, exist_ok=True)
acc_homog_df.to_csv(plot_data_loc + f'acc_homog-{sim_type}.csv',index=False)


def count_bin_pols_by_cluster(clusters):
    cons_agents = [agent for i,agent in enumerate(clusters) if i<len(clusters)//2]
    lib_agents = [agent for i,agent in enumerate(clusters) if i>=len(clusters)//2]
    cons_clusters, cons_counts = np.unique(cons_agents,return_counts=True)
    lib_clusters, lib_counts = np.unique(lib_agents,return_counts=True)

    return {'cons_clusters': dict(zip(cons_clusters,cons_counts)), 'lib_clusters': dict(zip(lib_clusters,lib_counts))}


def cluster_proportions(cluster_counts):
    cons_clusters = cluster_counts['cons_clusters']
    lib_clusters = cluster_counts['lib_clusters']
    all_cluster = set([cluster for cluster,_ in cons_clusters.items()] + [cluster for cluster,_ in lib_clusters.items()])

    cons_proportions = []
    # lib_proportions = []
    cluster_sizes = []
    n_agents = [sum(cons_clusters.values()), sum(lib_clusters.values())]
    for cluster in all_cluster:
        cluster_size = cons_clusters.get(cluster,0) + lib_clusters.get(cluster,0)
        cluster_sizes.append(cluster_size)

        cons_prop = (cons_clusters.get(cluster,0) / cluster_size)
        cons_proportions.append(cons_prop)

        # lib_prop = (lib_clusters.get(cluster,0) / cluster_size)
        # lib_proportions.append(lib_prop)

    # return {'cons_prop': cons_proportions, 'cluster_size': cluster_sizes}
    return list(zip(cons_proportions,cluster_sizes))

def proportion_correlations(clusters_df):
    cluster_props = []
    corrs = np.zeros(n_steps)
    for i,step in enumerate(np.arange(n_steps)*10):


        cons_props = clusters_df[clusters_df.step==step].clusters.apply(count_bin_pols_by_cluster).apply(cluster_proportions)
        merged_tuples = [tuple(item) for sublist in cons_props for item in sublist]

        cluster_props_df = pd.DataFrame(merged_tuples,columns=['cons_prop','cluster_size'])
        cluster_props.append(cluster_props_df)
        corrs[i] = cluster_props_df.cons_prop.corr(cluster_props_df.cluster_size)
    return corrs , cluster_props


cons_props = proportion_correlations(belief_clusters)


cons_corr = pd.DataFrame([10*np.arange(n_steps),cons_props[0]]).T
cons_corr.columns= ['step', 'cons_prop_corr']
cons_corr.to_csv(plot_data_loc + f'cons_prop_corr-{sim_type}.csv',index=False)


loc = 85
cons_props[1][loc].to_csv(plot_data_loc + f'size_prop-{sim_type}.csv',index=False)
