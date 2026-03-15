import os
import re
import torch
import random
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader.dataloader import Collater
from itertools import repeat, chain

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(smiles):
    # atoms
    mol = Chem.MolFromSmiles(smiles)
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

import pandas as pd
import numpy as np
import os 
import csv
import shutil

dict_path = "../dicts/"
data_path = "../../OOD/"
dataset_name = "ChChMiner_scaffold/"

if os.path.exists(data_path + dataset_name):
    shutil.rmtree(data_path + dataset_name)

cancer_drug = pd.read_csv(dict_path + "propertyDrugbank.csv")
drug_dict = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['SMILES']))
drug_dict_property = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['description']))
drug_dict_name = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['Common name']))

data = pd.read_csv(data_path + dataset_name[:-1] + "_test.csv")
data = np.array(data)
idx = np.random.permutation(len(data))
test_idx = idx

for i in range(len(test_idx)):
    j = test_idx[i]
    print(i)
    os.makedirs(data_path + dataset_name + "test/graph1/" + str(i))
    os.makedirs(data_path + dataset_name + "test/graph2/" + str(i))
    os.makedirs(data_path + dataset_name + "test/smiles1/" + str(i))
    os.makedirs(data_path + dataset_name + "test/smiles2/" + str(i))
    os.makedirs(data_path + dataset_name + "test/text/" + str(i))
    os.makedirs(data_path + dataset_name + "test/property1/" + str(i))
    os.makedirs(data_path + dataset_name + "test/property2/" + str(i))
    
    data1 = mol_to_graph_data_obj_simple(data[j][2])
    torch.save(data1, data_path + dataset_name + "test/graph1/" + str(i) + '/graph_data.pt')
    data2 = mol_to_graph_data_obj_simple(data[j][3])
    torch.save(data2, data_path + dataset_name + "test/graph2/" + str(i) + '/graph_data.pt')
    
    file = open(data_path + dataset_name + "test/text/" + str(i) + "/text.txt","w")
    label = data[j][4]
    if label == 0:
        text = "No."
    elif label == 1:
        text = "Yes."
    file.write(text + '\n')
    file.close()
    smiles1 = data[j][2]
    smiles2 = data[j][3]
    file = open(data_path + dataset_name + "test/smiles1/" + str(i) + "/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open(data_path + dataset_name + "test/smiles2/" + str(i) + "/text.txt","w")
    file.write(smiles2)
    file.close()

    file = open(data_path + dataset_name + "test/property1/" + str(i) + "/text.txt","w")
    property1 = "#Drug1 is " + drug_dict_name[data[j][0]] + ". "
    property1 += "" if drug_dict_property[data[j][0]].strip().endswith("Overview") else drug_dict_property[data[j][0]].strip()
    file.write(property1)
    file.close()
    file = open(data_path + dataset_name + "test/property2/" + str(i) + "/text.txt","w")
    property2 = "#Drug2 is " + drug_dict_name[data[j][1]] + ". "
    property2 += "" if drug_dict_property[data[j][1]].strip().endswith("Overview") else drug_dict_property[data[j][1]].strip()
    file.write(property2)
    file.close()
    i += 1

data = pd.read_csv(data_path + dataset_name[:-1] + "_train.csv")
data = np.array(data)
idx = np.random.permutation(len(data))
train_idx = idx

for i in range(len(train_idx)):
    j = train_idx[i]
    print(i)
    os.makedirs(data_path + dataset_name + "train/graph1/" + str(i))
    os.makedirs(data_path + dataset_name + "train/graph2/" + str(i))
    os.makedirs(data_path + dataset_name + "train/smiles1/" + str(i))
    os.makedirs(data_path + dataset_name + "train/smiles2/" + str(i))
    os.makedirs(data_path + dataset_name + "train/text/" + str(i))
    os.makedirs(data_path + dataset_name + "train/property1/" + str(i))
    os.makedirs(data_path + dataset_name + "train/property2/" + str(i))
    
    data1 = mol_to_graph_data_obj_simple(data[j][2])
    torch.save(data1, data_path + dataset_name + "train/graph1/" + str(i) + '/graph_data.pt')
    data2 = mol_to_graph_data_obj_simple(data[j][3])
    torch.save(data2, data_path + dataset_name + "train/graph2/" + str(i) + '/graph_data.pt')
    
    file = open(data_path + dataset_name + "train/text/" + str(i) + "/text.txt","w")
    label = data[j][4]
    if label == 0:
        text = "No."
    elif label == 1:
        text = "Yes."
    file.write(text + '\n')
    file.close()
    smiles1 = data[j][2]
    smiles2 = data[j][3]
    file = open(data_path + dataset_name + "train/smiles1/" + str(i) + "/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open(data_path + dataset_name + "train/smiles2/" + str(i) + "/text.txt","w")
    file.write(smiles2)
    file.close()

    file = open(data_path + dataset_name + "train/property1/" + str(i) + "/text.txt","w")
    property1 = "#Drug1 is " + drug_dict_name[data[j][0]] + ". "
    property1 += "" if drug_dict_property[data[j][0]].strip().endswith("Overview") else drug_dict_property[data[j][0]].strip()
    file.write(property1)
    file.close()
    file = open(data_path + dataset_name + "train/property2/" + str(i) + "/text.txt","w")
    property2 = "#Drug2 is " + drug_dict_name[data[j][1]] + ". "
    property2 += "" if drug_dict_property[data[j][1]].strip().endswith("Overview") else drug_dict_property[data[j][1]].strip()
    file.write(property2)
    file.close()
    i += 1

data = pd.read_csv(data_path + dataset_name[:-1] + "_valid.csv")
data = np.array(data)
idx = np.random.permutation(len(data))
valid_idx = idx

for i in range(len(valid_idx)):
    j = valid_idx[i]
    print(i)
    os.makedirs(data_path + dataset_name + "valid/graph1/" + str(i))
    os.makedirs(data_path + dataset_name + "valid/graph2/" + str(i))
    os.makedirs(data_path + dataset_name + "valid/smiles1/" + str(i))
    os.makedirs(data_path + dataset_name + "valid/smiles2/" + str(i))
    os.makedirs(data_path + dataset_name + "valid/text/" + str(i))
    os.makedirs(data_path + dataset_name + "valid/property1/" + str(i))
    os.makedirs(data_path + dataset_name + "valid/property2/" + str(i))
    
    data1 = mol_to_graph_data_obj_simple(data[j][2])
    torch.save(data1, data_path + dataset_name + "valid/graph1/" + str(i) + '/graph_data.pt')
    data2 = mol_to_graph_data_obj_simple(data[j][3])
    torch.save(data2, data_path + dataset_name + "valid/graph2/" + str(i) + '/graph_data.pt')
    
    file = open(data_path + dataset_name + "valid/text/" + str(i) + "/text.txt","w")
    label = data[j][4]
    if label == 0:
        text = "No."
    elif label == 1:
        text = "Yes."
    file.write(text + '\n')
    file.close()
    smiles1 = data[j][2]
    smiles2 = data[j][3]
    file = open(data_path + dataset_name + "valid/smiles1/" + str(i) + "/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open(data_path + dataset_name + "valid/smiles2/" + str(i) + "/text.txt","w")
    file.write(smiles2)
    file.close()

    file = open(data_path + dataset_name + "valid/property1/" + str(i) + "/text.txt","w")
    property1 = "#Drug1 is " + drug_dict_name[data[j][0]] + ". "
    property1 += "" if drug_dict_property[data[j][0]].strip().endswith("Overview") else drug_dict_property[data[j][0]].strip()
    file.write(property1)
    file.close()
    file = open(data_path + dataset_name + "valid/property2/" + str(i) + "/text.txt","w")
    property2 = "#Drug2 is " + drug_dict_name[data[j][1]] + ". "
    property2 += "" if drug_dict_property[data[j][1]].strip().endswith("Overview") else drug_dict_property[data[j][1]].strip()
    file.write(property2)
    file.close()
    i += 1
