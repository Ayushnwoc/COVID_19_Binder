import os
import sys
import argparse
from Bio.SeqIO import parse
import networkx as nx
from networkx.readwrite.gpickle import read_gpickle
import rdkit.Chem as Chem
import ccbmlib.models as ccbm
from ssw_aligner import local_pairwise_align_ssw
import pandas as pd
import numpy as np
import config

# The dataset to be used for the knowledge graph
pred_dataset = config.pred_dataset

# Threshold value for creating edges
threshold_drug_drug = 0.7
threshold_protein_protein = 0.7

# Get the command line arguments
parser = argparse.ArgumentParser("Python script to create knowledge graph from the davis dataset.")
parser.add_argument('--tdd', type=float, default=threshold_drug_drug, help="Threshold value for creating edges between two drugs")
parser.add_argument('--tpp', type=float, default=threshold_protein_protein, help="Threshold value for creating edges between two proteins")
args = parser.parse_args()

# Assign the arguments to the respective variables
threshold_drug_drug = args.tdd
threshold_protein_protein = args.tpp

print('\nConfiguration:')
print(f'Threshold value for creating edges between two drugs = {threshold_drug_drug}')
print(f'Threshold value for creating edges between two proteins = {threshold_protein_protein}')

# Read the smiles from the prediction dataset
smiles = []
for file in os.listdir('data/'+pred_dataset+'/smiles'):

    if file.endswith('.smi'):
        with open('data/'+pred_dataset+'/smiles/'+file, 'r') as f:
            data = f.read()

        temp = data.split('\n')[1:]
        for i in range(len(temp)):
            temp[i] = temp[i].split(' ')[0]

        for i in range(len(temp)):
            if(len(temp[i])>4):
                smiles.append(temp[i].upper())

# Read the sequences from the prediction dataset
protein_names = []
protein_seqs = []
proteins = {}
for file in os.listdir('data/'+pred_dataset+'/seq'):

    if file.endswith('.fasta'):
        with open('data/'+pred_dataset+'/seq/'+file, 'r') as f:
            records = parse(f, "fasta")
            for record in records:
                protein_names.append(str(record.name))
                protein_seqs.append(str(record.seq))
                proteins[str(record.name)] = str(record.seq)

# Remove the duplicate smiles
smiles = list(set(smiles))

# Remove the duplicate protein names
protein_names = list(set(protein_names))

# Read the graph
if os.path.isfile('KG.gpickle'):
    print('Loading pre-processed knowledge graph...')
    G = nx.read_gpickle('KG.gpickle')
else:
    print('Prepare the knowledge graph from the dataset first!')
    sys.exit()

# Get the proteins and drugs from the knowledge graph
smiles_KG = []
protein_names_KG = []
protein_seqs_KG = []
proteins_KG = {}
for node in G.nodes(data=True):
    if node[1]['type'] == 'drug':
        smiles_KG.append(node[0])
    else:
        protein_names_KG.append(node[0])
        protein_seqs_KG.append(node[1]['seq'])
        proteins_KG[node[0]] = node[1]['seq']

#print(f'\nKnowledge graph dataset: {dataset}')
print(f'Number of drugs: {len(smiles_KG)}')
print(f'Number of proteins: {len(proteins_KG)}')
print(f'Number of nodes: {G.number_of_nodes()}')
print(f'Number of edges: {G.number_of_edges()}\n')

print(f'\nPrediction dataset: {pred_dataset}')
print(f'Number of drugs: {len(smiles)}')
print(f'Number of proteins: {len(proteins)}\n')

# # Function to get Tanimoto Coefficient from smiles
# def tc_from_smiles(smile1, smile2):
#     mol1 = Chem.MolFromSmiles(smile1)
#     mol2 = Chem.MolFromSmiles(smile2)
#     maccs1 = ccbm.maccs_keys(mol1)
#     maccs2 = ccbm.maccs_keys(mol2)
#     maccs_tc = ccbm.tc(maccs1, maccs2)
#     return maccs_tc

# # Function to get alignment scores from protein sequences
# def alignment_score_from_seqs(seq1, seq2):
#     alignment = local_pairwise_align_ssw(seq1, seq2)
#     return alignment.optimal_alignment_score

# Function to get maccs from smiles
def maccs_from_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    maccs = ccbm.maccs_keys(mol)
    return maccs
    
print('Adding new drugs and proteins from the prediction dataset...')
print('Calculating Tanimoto Coefficients and creating edges...')
sms = []
sms_KG = []
tcs = []
# Get the Tanimoto Coefficients for drug-drug pairs from the knowledge graph and the prediction dataset and if it exceeds the threshold value, create an edge
for i in range(len(smiles)):
    try:
        maccs_new = maccs_from_smiles(smiles[i])
    except:
        continue
    for j in range(len(smiles_KG)):
        sms.append(smiles[i])
        sms_KG.append(smiles_KG[j])
        tc = ccbm.tc(maccs_new, G.nodes[smiles_KG[j]]['maccs'])
        tcs.append(tc)
        if tc > threshold_drug_drug:
            if not smiles[i] in G.nodes():
                G.add_node(smiles[i], maccs=maccs_new, type='drug_gen')
            G.add_edge(smiles[i], smiles_KG[j], weight=tc)

dict = {'smiles': sms, 'smiles_KG': sms_KG, 'tc': tcs}
df = pd.DataFrame(dict)
df.to_csv('TCs.csv')

print('Calculating alignment scores and creating edges...')
# Get the alignment scores for all protein-protein pairs from the knowledge graph and the prediction dataset and if it exceess the threshold value, create an edge
scores = []
prots = []
prots_KG = []
norm_scores = []
for i in range(len(protein_names)):
    for j in range(len(protein_names_KG)):
        score = local_pairwise_align_ssw(proteins[protein_names[i]], proteins_KG[protein_names_KG[j]]).optimal_alignment_score
        scores.append(score)

mean_score = np.mean(scores)
stdev_score = np.std(scores)
ctr=0
for i in range(len(protein_names)):
    for j in range(len(protein_names_KG)):
        score = (scores[ctr]-mean_score)/stdev_score
        ctr+=1
        prots.append(protein_names[i])
        prots_KG.append(protein_names_KG[j])
        norm_scores.append(score)
        if score > threshold_protein_protein:
            if not protein_names[i] in G.nodes():
                G.add_node(protein_names[i], seq=proteins[protein_names[i]], type='protein')
            G.add_edge(protein_names[i], protein_names_KG[j], weight=score)

dict = {"proteins": prots, "proteins_KG":prots_KG, 'scores': norm_scores}
df = pd.DataFrame(dict)
df.to_csv("alignment_scores.csv")

def n_neighbor(G, source, n):
    nodes = [source]
    node_visited = set()
    neighbors = []

    while n!=0:
        neighbors = []
        for node in nodes:
            node_visited.add(node)
            neighbors += [id for id in G.neighbors(node) if id not in node_visited]
        nodes = neighbors
        n-=1

    return neighbors

print('Generating final list of proteins and smiles...')
# Getting the list of generated molecules 3 hops away from the prediction proteins
smiles_final = {}
for name in protein_names:
    if name in G.nodes():
        nodes = n_neighbor(G, name, 3)
        temp = set()
        for node in nodes:
            if G.nodes[node]['type'] == 'drug_gen':
                temp.add(node)
        smiles_final[name] = list(temp)


# Save the list of proteins and corresponding drugs to be predicted for DTA in a csv file
dict = {}
compound_iso_smiles = []
target_name = []
target_sequence = []
for key in smiles_final.keys():
    for drug in smiles_final[key]:
        target_name.append(key)
        target_sequence.append(G.nodes[key]['seq'])
        compound_iso_smiles.append(drug)

print(f'\nNumber of drugs after filtering: {len(set(compound_iso_smiles))}\n')
dict = {'compound_iso_smiles': compound_iso_smiles, 'target_name':target_name, 'target_sequence': target_sequence}

file_path = 'data/'+pred_dataset+'/split/'+pred_dataset+'.csv'
df = pd.DataFrame(dict)
df.to_csv(file_path)