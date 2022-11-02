import copy
import os
import pandas as pd
import pickle
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.data import Batch

from graph_conversion import *
from metrics import *
from plot_losses import plot_losses
from plot_test import plot_test
from models.GEFA import GEFA
from models.GLFA import GLFA
import matplotlib.pyplot as plt

model_path = config.model_path

plot_drug = config.to_plot_drug
plot_prot = config.to_plot_prot

# Dataset on which model was trained
pred_dataset = config.pred_dataset
print('Dataset: ', pred_dataset)

# Model to be used
modeling = [GEFA, GLFA][config.run_model]
model_st = modeling.__name__

# GPU used
cuda_name = "cuda:" + str(config.cuda)
print('CUDA name:', cuda_name)


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB

class GraphPairDataset(Dataset):
    def __init__(self, smile_list, prot_list, dta_graph):
        self.smile_list = smile_list
        self.prot_list = prot_list
        self.dta_graph = dta_graph

    def __len__(self):
        return len(self.smile_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smile = self.smile_list[idx]
        prot = self.prot_list[idx]
        GCNData_Prot, GCNData_Smile = self.dta_graph[(prot, smile)]
        return GCNData_Smile, GCNData_Prot

# Function for prediction 
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            drug = data[0].to(device)
            prot = data[1].to(device)
            output = model(drug, prot)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_preds.numpy().flatten()

compound_iso_smiles = []
pdbs = []
pdbs_seqs = []

df = pd.read_csv('data/'+pred_dataset+'/split/'+pred_dataset+'.csv')
compound_iso_smiles += list(df['compound_iso_smiles'])
pdbs += list(df['target_name'])
pdbs_seqs += list(df['target_sequence'])
pdbs_tseqs = set(zip(pdbs, pdbs_seqs, compound_iso_smiles))


# Drugs pre-processing 
print('Pre-processing smiles...')
saved_drug_graph = {}
if os.path.isfile('saved_drug_graph_'+pred_dataset+'.pickle'):
    print('Load pre-processed file for drug_graph')
    with open('saved_drug_graph_'+pred_dataset+'.pickle', 'rb') as handle:
        saved_drug_graph = pickle.load(handle)

else:
    for smiles in compound_iso_smiles:
        c_size2, features2, edge_index2 = smile_to_graph(smiles, smiles + '.jpg', plot_drug)
        try:
            g2 = DATA.Data(
                x = torch.Tensor(features2),
                edge_index = torch.LongTensor(edge_index2).transpose(1, 0),
                )
        except:
            continue
        saved_drug_graph[smiles] = g2
    with open('saved_drug_graph_'+pred_dataset+'.pickle', 'wb') as handle:
        pickle.dump(saved_drug_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Protein pre-processing
dta_graph = {}
print('Pre-processing protein...')
saved_prot_graph = {}
if os.path.isfile('saved_prot_graph_'+pred_dataset+'.pickle'):
    print('Load pre-processed file for protein graph')
    with open('saved_prot_graph_'+pred_dataset+'.pickle', 'rb') as handle:
        saved_prot_graph = pickle.load(handle)

else:
    for target, seq in set(zip(pdbs, pdbs_seqs)):
        if os.path.isfile('data/' + pred_dataset + '/map/' + target + '.npy'):
            contactmap = np.load('data/' + pred_dataset + '/map/' + target + '.npy')
        else:
            raise FileNotFoundError
        c_size, features, edge_index, edge_weight = prot_to_graph(seq, contactmap, target, target + '.jpg', plot_prot, pred_dataset)
        g = DATA.Data(
            x = torch.Tensor(features),
            edge_index = torch.LongTensor(edge_index).transpose(1, 0),
            edge_attr = torch.FloatTensor(edge_weight),
            prot_len = c_size
        )
        saved_prot_graph[target] = g
    with open('saved_prot_graph_'+pred_dataset+'.pickle', 'wb') as handle:
        pickle.dump(saved_prot_graph, handle, protocol = pickle.HIGHEST_PROTOCOL)

print('Extracting from pre-processed Data...')
print(len(pdbs_tseqs))
index = 1
temp_compound_iso_smiles = []
temp_pdbs = []
temp_pdbs_seqs = []
for target, seq, smile in pdbs_tseqs:
    print('Extracting', index, '...')
    index+=1
    g = copy.deepcopy(saved_prot_graph[target])
    try:
        g2 = copy.deepcopy(saved_drug_graph[smile])    
    except:
        continue
    temp_compound_iso_smiles.append(smile)
    temp_pdbs.append(target)
    temp_pdbs_seqs.append(seq)
    dta_graph[(target, smile)] = [g, g2]
    num_feat_xp = g.x.size()[1]
    num_feat_xd = g2.x.size()[1]
    print("xp:", num_feat_xp, "| xd:", num_feat_xd)

compound_iso_smiles = temp_compound_iso_smiles
pdbs = temp_pdbs
pdbs_seqs = temp_pdbs_seqs
del(temp_compound_iso_smiles)
del(temp_pdbs)
del(temp_pdbs_seqs)

pred_drugs, pred_prots, pred_prots_seq = np.asarray(compound_iso_smiles), np.asarray(pdbs), np.asarray(pdbs_seqs)
pred_data = GraphPairDataset(smile_list=pred_drugs, dta_graph=dta_graph, prot_list=pred_prots)
pred_loader = DataLoader(pred_data, batch_size=config.PRED_BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

model = modeling(num_features_xd=num_feat_xd, num_features_xt=num_feat_xp, device=device).to(device)

prediction_file_name = 'data/' + pred_dataset + '/result/' + pred_dataset + '.csv'
model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'], strict=False)

P = predicting(model, device, pred_loader)
dict = {'compound_iso_smiles': compound_iso_smiles, 'target_name': pdbs, 'target_sequence': pdbs_seqs, 'predicted_affinity': P}
df = pd.DataFrame(dict)
df.to_csv(prediction_file_name)
