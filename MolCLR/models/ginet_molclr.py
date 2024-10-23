import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119 # valid: 1 to 118, including the extra mask tokens (0)
num_chirality_tag = 8 # 0 is the extra mask token
num_atom_degree = 12 # valid: 0 to 10, including the extra mask tokens (11)
num_atom_formal_charge = 13 # valid: -5 to 6 , including the extra mask tokens (7)
num_atom_hs = 10 # valid: 0 to 8 , including the extra mask tokens (9)
num_atom_radical_electrons = 6 # valid: 0 to 4 , including the extra mask tokens (5)
num_hybridization_type = 7 # 0 is the extra mask token
num_aromatic = 3 # valid: 0 to 1 , including the extra mask tokens (2)
num_ring = 3 # valid: 0 to 1 , including the extra mask tokens (2)

num_bond_type = 23 # including aromatic and self-loop edge (22)
num_stereo = 6
num_conjugated = 2


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_stereo, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_conjugated, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 3)
        self_loop_attr[:,0] = 22 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)


        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)


        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding3 = nn.Embedding(num_atom_degree, emb_dim)
        self.x_embedding4 = nn.Embedding(num_atom_formal_charge, emb_dim)
        self.x_embedding5 = nn.Embedding(num_atom_hs, emb_dim)
        self.x_embedding6 = nn.Embedding(num_atom_radical_electrons, emb_dim)
        self.x_embedding7 = nn.Embedding(num_hybridization_type, emb_dim)
        self.x_embedding8 = nn.Embedding(num_aromatic, emb_dim)
        self.x_embedding9 = nn.Embedding(num_ring, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        nn.init.xavier_uniform_(self.x_embedding9.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        else:
            pass
        
        if (pool == 'mean') or (pool == 'max') or (pool == 'add'):

            self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

            self.out_lin = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, self.feat_dim//2)
            )
        else:
            pass

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5]) + self.x_embedding7(x[:,6]) + self.x_embedding8(x[:,7]) + self.x_embedding9(x[:,8])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            
        if (self.pool == 'mean') or (self.pool == 'max') or (self.pool == 'add'):
            h = self.pool(h, data.batch)
            h = self.feat_lin(h)
            out = self.out_lin(h)
        else:
            out = None

        return h, out
