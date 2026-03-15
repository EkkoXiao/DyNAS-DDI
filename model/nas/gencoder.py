import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import add_self_loops,remove_self_loops
# from operations import *
from torch.autograd import Variable
# from genotypes import Genotype
from torch_geometric.nn import  global_mean_pool,global_add_pool
from model.nas.chems import AtomEncoder
from torch.nn.utils.rnn import pad_sequence

from autogllight.nas.space.graces_space.op_graph_classification import Pooling_trivial, NA_OPS, Readout_trivial

class NaSingleOp(nn.Module):
  def __init__(self, in_dim, out_dim, with_linear):
    super().__init__()
    self._ops = nn.ModuleList()

    self.op = NA_OPS['gin'](in_dim, out_dim)
    if with_linear:
      self.op_linear = torch.nn.Linear(in_dim, out_dim)

  def forward(self, x, edge_index, edge_weights, edge_attr, with_linear):
    mixed_res = []
    if with_linear:
        mixed_res.append(self.op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr)+self.op_linear(x))
    else:
        mixed_res.append(self.op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr))
    return sum(mixed_res)

class NaDisenOp(nn.Module):
  def __init__(self, in_dim, out_dim, with_linear, k = 4):
    super().__init__()
    self.ops = nn.ModuleList()
    self.op_linear = nn.ModuleList()
    self.in_dim = in_dim
    self.k = k

    for i in range(k):
      self.ops.append(NA_OPS['gin'](in_dim // 4, out_dim // 4))
      if with_linear:
        self.op_linear.append(torch.nn.Linear(in_dim, out_dim))

  def forward(self, x, edge_index, edge_weights, edge_attr, with_linear):
    # x: node * d
    mixed_res = []
    xs = x.hsplit(self.k)
    for i in range(self.k):
      z = self.ops[i](xs[i], edge_index, edge_weight=edge_weights, edge_attr=edge_attr)
      if with_linear:
        z = z + self.op_linear[i](xs[i])
      mixed_res.append(z)
    res = torch.hstack(mixed_res)
    return res

class Disen3Head(nn.Module):
  def __init__(self, in_dim, k = 4):
    super().__init__()
    self.ops = nn.ModuleList()
    self.in_dim = in_dim
    self.k = k

    for i in range(3):
      self.ops.append(torch.nn.Linear(in_dim // 4, 1))

  def forward(self, x):
    # x: node * d
    mixed_res = []
    xs = x.hsplit(self.k)
    for i in range(3):
      z = self.ops[i](xs[i])
      z = 0.05 + 0.35 * torch.sigmoid(z)
      mixed_res.append(z)
    res = torch.hstack(mixed_res)
    return res

class GEncoder(nn.Module):
  def __init__(self, in_dim, hidden_size, num_layers=2, dropout=0.5, epsilon=0.0, args=None, with_conv_linear=False, mol = False, virtual = False):
    super().__init__()

    self.in_dim = in_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.epsilon = epsilon
    self.with_linear = with_conv_linear
    self.explore_num = 0
    self.args = args
    self.temp = args.temp
    self._loc_mean = args.loc_mean
    self._loc_std = args.loc_std
    self.mol = mol # if the task is molecule
    self.virtual = virtual
    self.use_att = args.use_att
    if not self.mol:
      self.lin1 = nn.Linear(in_dim, hidden_size)
    else:
      self.atom_encoder = AtomEncoder(hidden_size)
    self.virtualnode_embedding = torch.nn.Embedding(1, hidden_size)
    torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

    self.mlp_virtualnode_list = torch.nn.ModuleList()
    for layer in range(num_layers - 1):
        self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden_size, 2*hidden_size), torch.nn.BatchNorm1d(2*hidden_size), torch.nn.ReLU(), \
                                                torch.nn.Linear(2*hidden_size, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU()))

    # node aggregator op
    self.gnn_layers = nn.ModuleList()
    for i in range(num_layers):
        if i < 1:
          self.gnn_layers.append(NaSingleOp(hidden_size, hidden_size, self.with_linear))
        else:
          self.gnn_layers.append(NaDisenOp(hidden_size, hidden_size, self.with_linear))

    self.pooling_trivial = Pooling_trivial(hidden_size * (num_layers + 1), args.pooling_ratio)

    self.layer7 = Readout_trivial()
    self.disenhead = Disen3Head(hidden_size)
    self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size * (num_layers + 1), num_heads=args.att_head, batch_first=True)

    self.lin_output = nn.Linear(hidden_size * 2 * (num_layers + 1), hidden_size)

  def encode(self, data):
    self.args.search_act = False
    with_linear = self.with_linear
    # Extract node features and edge information from input data
    x, edge_index = data.x, data.edge_index
    edge_attr = getattr(data, 'edge_attr', None)
    batch = data.batch
    # If edge attributes are missing, add self-loops to ensure each node has at least one edge
    if edge_attr == None:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size()[0])
    # Try applying a linear transform to input features; otherwise use the dedicated encoder
    if not self.mol:
        x = F.elu(self.lin1(x))
    else:
        x = self.atom_encoder(x)
    # Initialize edge weights to 1 if not provided
    edge_weights = torch.ones(edge_index.size()[1], device=edge_index.device).float()
    # Initialize virtual node embeddings to enhance global features
    virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
    # Store node embeddings from each layer
    gr = [x]

    for i in range(self.num_layers):
        # If virtual nodes are enabled, add their embeddings to node features
        if self.virtual:
            orix = x
            x = x + virtualnode_embedding[batch]
        # Update node embeddings using graph neural network layers
        x = self.gnn_layers[i](x, edge_index, edge_weights, edge_attr, with_linear)
        x = F.elu(x)
        # Normalize node embeddings at each layer
        layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
        x = layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        gr.append(x)
        # Update virtual node embeddings
        if self.virtual and i < self.num_layers - 1:
            virtualnode_embedding_temp = global_add_pool(orix, batch) + virtualnode_embedding
            virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.dropout, training = self.training)
    # Concatenate node embeddings from all layers
    gr = torch.cat(gr, 1)
    return gr, edge_index, edge_weights, batch
     
  def embed(self, data, edge_index, edge_weights, batch, x):
    x, edge_index, edge_weights, batch, _ = self.pooling_trivial(x, edge_index, edge_weights, data, batch, None)

    # Obtain graph embeddings through global pooling
    x5 = self.layer7(x, batch)
    # Further process graph embeddings with linear layers and nonlinear activations
    x5 = self.lin_output(x5)
    x5 = F.elu(x5)
    return x5
  
  def attention(self, query, key, query_batch, key_batch):
    B = int(query_batch.max().item()) + 1
    # Group nodes by batch
    query_list = [query[query_batch == i] for i in range(B)]
    key_list    = [key[key_batch == i] for i in range(B)]
    # Pad each group to the same length
    queries = pad_sequence(query_list, batch_first=True)  
    keys   = pad_sequence(key_list, batch_first=True)      
    # Build mask: True indicates a padded position
    key_lengths = torch.tensor([x.size(0) for x in key_list], device=keys.device)
    key_mask = torch.arange(keys.size(1), device=keys.device).expand(B, keys.size(1)) >= key_lengths.unsqueeze(1)

    query_lengths = torch.tensor([x.size(0) for x in query_list], device=queries.device)
    query_mask = torch.arange(queries.size(1), device=queries.device).expand(B, queries.size(1)) >= query_lengths.unsqueeze(1)

    attn_query, _ = self.multihead_attn(query=queries,
                                        key=keys,
                                        value=keys,
                                        key_padding_mask=key_mask)
    # Residual connection update
    queries = queries + attn_query
    attn_key, _ = self.multihead_attn(query=keys,
                                        key=queries,
                                        value=queries,
                                        key_padding_mask=query_mask)
    # Residual connection update
    keys = keys + attn_key
    return torch.cat([queries[i, :query_list[i].size(0), :] for i in range(B)], dim=0), torch.cat([keys[i, :key_list[i].size(0), :] for i in range(B)], dim=0)

  def forward(self, data1, data2, discrete=False, mode='none'):
    gr1, edge_index1, edge_weights1, batch1 = self.encode(data1)
    # Reduce graph complexity via pooling operations
    x_emb1 = self.embed(data1, edge_index1, edge_weights1, batch1, gr1)
    # Use a decoupled head to generate auxiliary task outputs (SSL)
    ssloutput1 = self.disenhead(x_emb1)

    gr2, edge_index2, edge_weights2, batch2 = self.encode(data1)
    # Reduce graph complexity via pooling operations
    x_emb2 = self.embed(data2, edge_index2, edge_weights2, batch2, gr2)
    # Use a decoupled head to generate auxiliary task outputs (SSL)
    ssloutput2 = self.disenhead(x_emb2)

    if self.use_att:
      att1, att2 = self.attention(gr1, gr2, batch1, batch2)
      emb1 = self.embed(data1, edge_index1, edge_weights1, batch1, att1)
      emb2 = self.embed(data2, edge_index2, edge_weights2, batch2, att2)

      return emb1, emb2, ssloutput1, ssloutput2
    else:
      return x_emb1, x_emb2, ssloutput1, ssloutput2

  def arch_parameters(self):
    return self._arch_parameters
