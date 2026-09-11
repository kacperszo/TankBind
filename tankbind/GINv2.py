# code from torchdrug.
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_ops import scatter_add, scatter_max, scatter_mean
from GATv2 import _ScatterReadout
from collections.abc import Sequence


class _MLP(nn.Module):
    """torchdrug's `layers.MLP`, reimplemented so this file needs no torchdrug.

    The module structure is kept identical (`self.layers` as a ModuleList of Linear), because
    the published checkpoint addresses these parameters by name — `mlp.layers.0.weight` and so
    on. A functionally equivalent MLP with different attribute names would simply fail to load.

    Signature order matters too, and is the reason for the note at the call site: the third
    positional argument is `short_cut`, not `activation`.
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False,
                 activation="relu", dropout=0):
        super().__init__()
        if not isinstance(hidden_dims, (list, tuple)):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation
        self.dropout = nn.Dropout(dropout) if dropout else None

        self.layers = nn.ModuleList(
            nn.Linear(self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1))
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(self.dims[i + 1]) for i in range(len(self.dims) - 2)
        ) if batch_norm else None

    def forward(self, input):
        layer_input = input
        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        return hidden


class GraphIsomorphismConv(nn.Module):
    """
    Graph isomorphism convolution operator from `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf
    
    torchdrug:
        https://torchdrug.ai/

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        hidden_dims (list of int, optional): hidden dimensions
        eps (float, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dims=None, eps=0, learn_eps=False,
                 batch_norm=False, activation="relu"):
        super(GraphIsomorphismConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        eps = torch.tensor([eps], dtype=torch.float32)
        if learn_eps:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if hidden_dims is None:
            hidden_dims = []
        # NOTE the third positional is `short_cut`, not `activation` — TankBind passes the
        # activation there, so a string like "relu" lands as a truthy short_cut while the
        # activation keeps its default. Reproduced deliberately: the published weights were
        # trained with residual connections in this MLP.
        self.mlp = _MLP(input_dim, list(hidden_dims) + [output_dim], activation)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, edge_list, edge_feature, input):
        #
        node_in = edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(edge_feature.float())
        return message


    def aggregate(self, edge_list, edge_weight, num_node, message):
        node_out = edge_list[:, 1]
        edge_weight = edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=num_node)
        return update


    def message_and_aggregate(self, edge_list, edge_weight, edge_feature, num_node, input):
        # torchdrug's sparse_coo_tensor is torch's with the index check skipped for speed;
        # the resulting tensor is the same, and this drops a JIT-compiled C++ extension.
        adjacency = torch.sparse_coo_tensor(edge_list.t()[:2], edge_weight,
                                            (num_node, num_node))
        update = torch.sparse.mm(adjacency.t().float(), input.float())
        if self.edge_linear:
            edge_input = edge_feature.float()
            edge_weight = edge_weight.unsqueeze(-1)
            if self.edge_linear.in_features > self.edge_linear.out_features:
                edge_input = self.edge_linear(edge_input)
            edge_update = scatter_add(edge_input * edge_weight, edge_list[:, 1], dim=0,
                                      dim_size=num_node)
            if self.edge_linear.in_features <= self.edge_linear.out_features:
                edge_update = self.edge_linear(edge_update)
            update += edge_update

        return update


    def combine(self, input, update):
        output = self.mlp((1 + self.eps) * input + update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, edge_list, edge_weight, edge_feature, num_node, input):
        """
        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        update = self.message_and_aggregate(edge_list, edge_weight, edge_feature, num_node, input)
        output = self.combine(input, update)
        return output

class GIN(nn.Module):
    """
    Graph Ismorphism Network proposed in `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf
    
    .. _torchdrug:
        https://torchdrug.ai/
    
    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        num_mlp_layer (int, optional): number of MLP layers
        eps (int, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim=None, hidden_dims=None, edge_input_dim=None, num_mlp_layer=2, eps=0, learn_eps=False,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False,
                 readout="sum"):
        super(GIN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            layer_hidden_dims = [self.dims[i + 1]] * (num_mlp_layer - 1)
            self.layers.append(GraphIsomorphismConv(self.dims[i], self.dims[i + 1], edge_input_dim,
                                                           layer_hidden_dims, eps, learn_eps, batch_norm, activation))

        # same substitution as GATv2: torchdrug's readouts are a scatter over the per-node
        # graph index, and `data`/`utils` were imported but unused.
        if readout in ("sum", "mean"):
            self.readout = _ScatterReadout(readout)
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, edge_list, edge_weight, edge_feature, num_node, input, all_loss=None, metric=None):
        """
        Compute the node representations.

        Parameters:
            
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` fields:
                node representations of shape :math:`(|V|, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(edge_list, edge_weight, edge_feature, num_node, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return {
            "node_feature": node_feature
        }