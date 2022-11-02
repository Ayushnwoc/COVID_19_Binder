import sys
import os
import argparse
import networkx as nx
import rdkit.Chem as Chem
import ccbmlib.models as ccbm
from ssw_aligner import local_pairwise_align_ssw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import pickle

# # The dataset to be used for the knowledge graph
# dataset = config.dataset

# Threshold value for creating edges
threshold_drug_drug = 0.7
threshold_protein_protein = 0.7
#threshold_drug_protein = 7.0

# Get the command line arguments
parser = argparse.ArgumentParser("Python script to create knowledge graph from the davis dataset.")
parser.add_argument('--tdd', type=float, default=threshold_drug_drug, help="Threshold value for creating edges between two drugs")
parser.add_argument('--tpp', type=float, default=threshold_protein_protein, help="Threshold value for creating edges between two proteins")
#parser.add_argument('--tdp', type=float, default=threshold_drug_protein, help="Threshold value for creating edges between a drug and a protein")
args = parser.parse_args()

# Assign the arguments to the respective variables
threshold_drug_drug = args.tdd
threshold_protein_protein = args.tpp
#threshold_drug_protein = args.tdp

print('\nConfiguration:')
print(f'Threshold value for creating edges between two drugs = {threshold_drug_drug}')
print(f'Threshold value for creating edges between two proteins = {threshold_protein_protein}')
#print(f'Threshold value for creating edges between a drug and a protein = {threshold_drug_protein}\n')

# Read the dataset for creating knowledge graph
davis_smiles = []
davis_protein_names = []
davis_protein_seqs = []
davis_affinity = []
for file in os.listdir('data/davis/split'):
    if file.endswith('.csv'):
        data = pd.read_csv('data/davis/split/'+file)
        davis_smiles += list(data['compound_iso_smiles'])
        davis_protein_names += list(data['target_name'])
        davis_protein_seqs += list(data['target_sequence'])
        davis_affinity += list(data['affinity'])

kiba_smiles = []
kiba_protein_names = []
kiba_protein_seqs = []
kiba_affinity = []
with open('data/kiba/split/kiba.pickle', 'rb') as f:
    data = pickle.load(f)
    kiba_smiles+=list(data['compound_iso_smiles'])
    kiba_protein_names += list(data['target_name'])
    kiba_protein_seqs += list(data['target_sequence'])
    kiba_affinity += list(data['affinity'])

smiles = davis_smiles + kiba_smiles
protein_names = davis_protein_names + kiba_protein_names
protein_seqs = davis_protein_seqs + kiba_protein_seqs

# Remove the duplicate smiles
smiles = list(set(smiles))

# Create a dictionary with the proteins names as keys and their sequences as the values
proteins = {}
for name, seq in zip(protein_names,  protein_seqs):
    proteins[name]  = seq

# Remove the duplicate protein names
protein_names = list(set(protein_names))

# print(f'Dataset: {dataset}')
print(f'Number of drugs: {len(smiles)}')
print(f'Number of proteins: {len(proteins)}\n')

# Function to get maccs from smiles
def maccs_from_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    maccs = ccbm.maccs_keys(mol)
    return maccs

# Create the graph
G = nx.Graph()

maccs_dict = {}  # Stores the maccs of proteins
print('Adding drugs and proteins to graph...')
print('Calculating Tanimoto Coefficients and creating edges...')
# Get the Tanimoto Coefficients for all drug-drug pairs from the dataset and if it exceeds the threshold value for a pair, create an edge
for i in range(len(smiles)):
    for j in range(i+1, len(smiles)):
        if i==0:
            if j==1:
                maccs_dict[smiles[i]] = maccs_from_smiles(smiles[i])
            maccs_dict[smiles[j]] = maccs_from_smiles(smiles[j])
        tc = ccbm.tc(maccs_dict[smiles[i]], maccs_dict[smiles[j]])
        if tc > threshold_drug_drug:
            if not smiles[i] in G.nodes():
                G.add_node(smiles[i], maccs=maccs_dict[smiles[i]], type='drug')
            if not smiles[j] in G.nodes():
                G.add_node(smiles[j], maccs=maccs_dict[smiles[j]], type='drug')
            G.add_edge(smiles[i], smiles[j], weight=tc)

print('Calculating alignment scores and creating edges...')
# Get the alignment scores for all protein-protein pairs from the data and if it exceeds the threshold value for a pair, create an edge
scores = []
for i in range(len(protein_names)):
    for j in range(i+1, len(protein_names)):
        score = local_pairwise_align_ssw(proteins[protein_names[i]], proteins[protein_names[j]]).optimal_alignment_score
        scores.append(score)

mean_score = np.mean(scores)
stdev_score = np.std(scores)
ctr=0
for i in range(len(protein_names)):
    for j in range(i+1, len(protein_names)):
        score = (scores[ctr]-mean_score)/stdev_score
        ctr+=1
        if score > threshold_protein_protein:
            if not protein_names[i] in G.nodes():
                G.add_node(protein_names[i], seq=proteins[protein_names[i]], type='protein')
            if not protein_names[j] in G.nodes():
                G.add_node(protein_names[j], seq=proteins[protein_names[j]], type='protein')
            G.add_edge(protein_names[i], protein_names[j], weight=score)

print('Reading DTA scores and creating edges...')
# Get the affinities for drug-protein pairs from the data and if it exceeds the threshold value for a pair, create an edge
for i in range(len(davis_smiles)):
    if(davis_affinity[i]>7.0):
        if not davis_smiles[i] in G.nodes():
            if not davis_smiles[i] in maccs_dict.keys():
                maccs_dict[davis_smiles[i]] = maccs_from_smiles(davis_smiles[i])
            G.add_node(davis_smiles[i], maccs=maccs_dict[davis_smiles[i]], type='drug')
        if not davis_protein_names[i] in G.nodes():
            G.add_node(davis_protein_names[i], seq=davis_protein_seqs[i], type='protein')
        G.add_edge(davis_smiles[i], davis_protein_names[i], weight = davis_affinity[i])

for i in range(len(kiba_smiles)):
    if(kiba_affinity[i]<=3.0):
        if not kiba_smiles[i] in G.nodes():
            if not kiba_smiles[i] in maccs_dict.keys():
                maccs_dict[kiba_smiles[i]] = maccs_from_smiles(kiba_smiles[i])
            G.add_node(kiba_smiles[i], maccs=maccs_dict[kiba_smiles[i]], type='drug')
        if not kiba_protein_names[i] in G.nodes():
            G.add_node(kiba_protein_names[i], seq=kiba_protein_seqs[i], type='protein')
        G.add_edge(kiba_smiles[i], kiba_protein_names[i], weight = kiba_affinity[i])

print(f'\nNumber of nodes: {G.number_of_nodes()}')
print(f'Number of edges: {G.number_of_edges()}\n')
print('Done! Saving the graph to '+os.getcwd()+'/KG.gpickle')
# Save the graph
nx.write_gpickle(G, "KG.gpickle")