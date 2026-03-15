import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import add_self_loops,remove_self_loops
# from operations import *
from .op_graph_classification import *
from torch.autograd import Variable
from .genotypes import NA_PRIMITIVES, LA_PRIMITIVES, POOL_PRIMITIVES, READOUT_PRIMITIVES, ACT_PRIMITIVES
# from genotypes import Genotype
from torch_geometric.nn import  global_mean_pool,global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder

class NaSingleOp(nn.Module):
  def __init__(self, in_dim, out_dim, with_linear):
    super().__init__()
    self._ops = nn.ModuleList()

    self.op = NA_OPS['gin'](in_dim, out_dim)

    self.op_linear = torch.nn.Linear(in_dim, out_dim)

  def forward(self, x, edge_index, edge_weights, edge_attr, with_linear):
    mixed_res = []
    if with_linear:
        mixed_res.append(self.op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr)+self.op_linear(x))
        # print('with linear')
    else:
        mixed_res.append(self.op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr))
        # print('without linear')
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
  def __init__(self, criterion, in_dim, out_dim, hidden_size, num_layers=2, dropout=0.5, epsilon=0.0, args=None, with_conv_linear=False,num_nodes=0, mol = False, virtual = False):
    super().__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_nodes = num_nodes
    self._criterion = criterion
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

    self.lin1 = nn.Linear(in_dim, hidden_size)
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

    self.lin_output = nn.Linear(hidden_size * 2 * (num_layers + 1), hidden_size)
    self.classifier = nn.Linear(hidden_size, out_dim)

  def forward(self, data, discrete=False, mode='none'):
    
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
    try:
        x = F.elu(self.lin1(x))
    except RuntimeError:
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
    # Reduce graph complexity via pooling operations
    x, edge_index, edge_weights, batch, _ = self.pooling_trivial(gr, edge_index, edge_weights, data, batch, None)
    # Obtain graph embeddings through global pooling
    x5 = self.layer7(x, batch)
    # Further process graph embeddings with linear layers and nonlinear activations
    x5 = self.lin_output(x5)
    x5 = F.elu(x5)
    # Obtain the final graph embedding for downstream tasks
    x_emb = x5
    # Use a decoupled head to generate auxiliary task outputs (SSL)
    ssloutput = self.disenhead(x_emb)
    # Get prediction results through the classifier
    logits = self.classifier(x5)

    if not self.mol:
        logits = F.log_softmax(logits, dim = -1)

    return logits, x_emb, ssloutput

  def arch_parameters(self):
    return self._arch_parameters
