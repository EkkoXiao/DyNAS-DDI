import torch
import torch_geometric
from torch_geometric.data import Dataset
import os
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import re
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

QUESTION_DDI = " Do the two drugs exhibit drug-drug interactions?"
QUESTION_DDI_MULTI = " What is the drug-drug interaction of the two drugs?"
QUESTION_PROPERTY = " What are the descriptions of the two drugs?"

def count_subdirectories(folder_path):
    try:
        entries = os.listdir(folder_path)
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
        return len(subdirectories)
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not exist.")
        return -1
    except Exception as e:
        print(f"Error: {e}")
        return -2 

def compute_motif_pagerank_tensor(smiles, k=3):
    """
    Compute a tensor of PageRank probabilities for the top-k motifs in a molecule.
    
    Args:
        smiles (str): The SMILES string of the molecule.
        k (int): The number of top motifs to extract (default: 3).
    
    Returns:
        np.array: A NumPy array containing PageRank probabilities for the selected motifs.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Perform BRICS fragmentation
    fragments = list(BRICS.BRICSDecompose(mol))
    fragments = [(frag, Chem.MolFromSmiles(frag).GetNumAtoms()) for frag in fragments]
    fragments.sort(key=lambda x: x[1], reverse=True)

    # Select the top-k motifs; if fewer than k, fill with "C" (single carbon)
    motifs = [fragments[i][0] if i < len(fragments) else "C" for i in range(k)]

    # Construct the molecular graph
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # Compute PageRank scores
    pagerank_scores = nx.pagerank(G)

    # Compute PageRank probabilities for each motif
    motif_probs = []
    for motif in motifs:
        motif_cleaned = re.sub(r"\[\d*\*\]", "", motif)  # Remove BRICS break marks
        motif_cleaned = re.sub(r"\(\)", "", motif_cleaned)
        motif_mol = Chem.MolFromSmiles(motif_cleaned, sanitize=False)
        if motif_mol is None:
            motif_probs.append(0.0)
            continue

        matches = mol.GetSubstructMatches(motif_mol)
        nodes = list(matches[0]) if matches else []
        motif_probs.append(sum(pagerank_scores[n] for n in nodes) if nodes else 0.0)

    return np.array(motif_probs, dtype=np.float32)

class DegreeDistribution(object):
    def __call__(self, g):
        edges = g.edge_index[1]
        if edges.numel() == 0:
            deratio = torch.tensor([0.0, 0.0, 0.0])
        else:
            degrees = torch_geometric.utils.degree(edges).to(torch.long).numpy().tolist()
            deratio = [degrees.count(i) for i in range(1, 4)]
            deratio = torch.tensor(deratio) / g.num_nodes
        g.deratio = deratio
        return g
    

class PropertyDistribution(object):
    def __call__(self, g, text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        deratio = torch.tensor([torch.max(cls_embedding).item(), torch.min(cls_embedding).item(), torch.mean(cls_embedding).item()])
        g.deratio = deratio
        return g
    

class MotifDistribution(object):
    def __call__(self, g, smiles):
        g.deratio = torch.tensor(compute_motif_pagerank_tensor(smiles, k=3))
        return g
    
class MoleculeDatasetDDI(Dataset):
    def __init__(self, root, text_max_len, prompt=None, transform=None):
        super(MoleculeDatasetDDI, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.transform = transform
        if self.transform == "Property":
            self.text_model = AutoModel.from_pretrained("bert_pretrained/")
            self.text_tokenizer = AutoTokenizer.from_pretrained("bert_pretrained/")
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return count_subdirectories(self.root+"text/")


    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles1_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles1_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            smiles1 = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles1[:128])
        else:
            smiles_prompt1 = self.prompt
        
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)

        smiles2_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles2_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            smiles2 = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles2[:128])
        else:
            smiles_prompt2 = self.prompt
            
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+ QUESTION_DDI
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)

        if self.transform == "Degree":
            data_graph1 = DegreeDistribution()(data_graph1)
            data_graph2 = DegreeDistribution()(data_graph2)
        elif self.transform == "Property":
            property1_name_list = os.listdir(self.root+'property1/'+str(index)+'/')
            property1_path = os.path.join(self.root, 'property1/'+str(index)+'/', property1_name_list[0])
            with open(property1_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                property1 = lines[0].strip()
            property2_name_list = os.listdir(self.root+'property2/'+str(index)+'/')
            property2_path = os.path.join(self.root, 'property2/'+str(index)+'/', property2_name_list[0])
            with open(property2_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                property2 = lines[0].strip()
            data_graph1 = PropertyDistribution()(data_graph1, property1, self.text_model, self.text_tokenizer)
            data_graph2 = PropertyDistribution()(data_graph2, property2, self.text_model, self.text_tokenizer)
        elif self.transform == "Motif":
            data_graph1 = MotifDistribution()(data_graph1, smiles1)
            data_graph2 = MotifDistribution()(data_graph2, smiles2)

        return data_graph1, data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token
    

class MoleculeDatasetDDIPretrain(Dataset):
    def __init__(self, root, text_max_len, prompt=None, transform=None, content=True):
        super(MoleculeDatasetDDIPretrain, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.transform = transform
        self.content = content
        if self.transform == "Property":
            self.text_model = AutoModel.from_pretrained("bert_pretrained/")
            self.text_tokenizer = AutoTokenizer.from_pretrained("bert_pretrained/")

        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        if self.content:
            return (count_subdirectories(self.root+"text/") * 2)
        else:
            return count_subdirectories(self.root+"text/")

    def __getitem__(self, index):
        idx = (index // 2) if self.content else index
        graph1_name_list = os.listdir(self.root+'graph1/'+str(idx)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(idx)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(idx)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(idx)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(idx)+'/')
        property1_name_list = os.listdir(self.root+'property1/'+str(idx)+'/')
        property2_name_list = os.listdir(self.root+'property2/'+str(idx)+'/')
        
        graph_path = os.path.join(self.root, 'graph1/'+str(idx)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles1_path = os.path.join(self.root, 'smiles1/'+str(idx)+'/', smiles1_name_list[0])
        with open(smiles1_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            smiles1 = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles1[:128])
        else:
            smiles_prompt1 = self.prompt
        
        graph_path = os.path.join(self.root, 'graph2/'+str(idx)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)

        smiles2_path = os.path.join(self.root, 'smiles2/'+str(idx)+'/', smiles2_name_list[0])
        with open(smiles2_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            smiles2 = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles2[:128])
        else:
            smiles_prompt2 = self.prompt

        if self.transform == "Degree":
            data_graph1 = DegreeDistribution()(data_graph1)
            data_graph2 = DegreeDistribution()(data_graph2)
        elif self.transform == "Property":
            property1_name_list = os.listdir(self.root+'property1/'+str(index)+'/')
            property1_path = os.path.join(self.root, 'property1/'+str(index)+'/', property1_name_list[0])
            with open(property1_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                property1 = lines[0].strip()[:128]
            property2_name_list = os.listdir(self.root+'property2/'+str(index)+'/')
            property2_path = os.path.join(self.root, 'property2/'+str(index)+'/', property2_name_list[0])
            with open(property2_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                property2 = lines[0].strip()[:128]
            data_graph1 = PropertyDistribution()(data_graph1, property1, self.text_model, self.text_tokenizer)
            data_graph2 = PropertyDistribution()(data_graph2, property2, self.text_model, self.text_tokenizer)
        elif self.transform == "Motif":
            data_graph1 = MotifDistribution()(data_graph1, smiles1)
            data_graph2 = MotifDistribution()(data_graph2, smiles2)

        if self.content and index % 2 == 1:
            smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+ QUESTION_PROPERTY
            property1_path = os.path.join(self.root, 'property1/'+str(idx)+'/', property1_name_list[0])
            with open(property1_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                property1 = lines[0].strip()[:128]
            property2_path = os.path.join(self.root, 'property2/'+str(idx)+'/', property2_name_list[0])
            with open(property2_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                property2 = lines[0].strip()[:128]
            return data_graph1, data_graph2, property1[:128] + ' ' + property2[:128], smiles_prompt
        else:
            smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+ QUESTION_DDI
            text_path = os.path.join(self.root, 'text/'+str(idx)+'/', text_name_list[0])
            
            with open(text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                text = lines[0].strip()
            return data_graph1, data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token