# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Callable, List

import gin
from torch_geometric.typing import Adj

import copy

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Tanh

from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv, GATConv, \
    SGConv, GATv2Conv, ARMAConv, FiLMConv, SuperGATConv, TransformerConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from torch_geometric.nn.conv import APPNP as APPNPConv
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from hgcn.models.encoders import HGCN as _HGCNEncoder
from hgcn.models.decoders import LinearDecoder

class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        self.convs = ModuleList()

        self.norms = None
        if norm is not None:
            self.norms = ModuleList(
                [copy.deepcopy(norm) for _ in range(num_layers)])

        if jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if out_channels is not None:
            self.out_channels = out_channels
            if jk == 'cat':
                self.lin = Linear(num_layers * hidden_channels, out_channels)
            else:
                self.lin = Linear(hidden_channels, out_channels)
        else:
            if jk == 'cat':
                self.out_channels = num_layers * hidden_channels
            else:
                self.out_channels = hidden_channels

        print(self)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(in_channels = {self.in_channels}, '
                f'hidden_channels = {self.hidden_channels}, '
                f'out_channels = {self.out_channels}, num_layers={self.num_layers}) ')


@gin.configurable
class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(GCNConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, **kwargs))

@gin.configurable
class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(SAGEConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, **kwargs))

@gin.configurable
class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(
            GINConv(GIN.MLP(in_channels, hidden_channels), **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GINConv(GIN.MLP(hidden_channels, hidden_channels), **kwargs))

    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

@gin.configurable
class GAT(BasicGNN):
    r"""The Graph Neural Network from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper, using the
    :class:`~torch_geometric.nn.GATConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            GATConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(GATConv(hidden_channels, out_channels, **kwargs))


@gin.configurable
class MLP(torch.nn.Module):
    r"""Multi-layer Perceptron.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True), **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        layers = []
        layers.append(Linear(in_channels, hidden_channels))
        layers.append(self.act)
        layers.append(torch.nn.Dropout(dropout))
        for i in range(1, num_layers):
            layers.append(Linear(hidden_channels, hidden_channels))
            layers.append(self.act)
            layers.append(torch.nn.Dropout(dropout))

        if out_channels is not None:
            layers.append(Linear(hidden_channels, out_channels))

        self.model = Sequential(*layers)
        print(self.__repr__())

    def reset_parameters(self):
        for module in self.model:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, batch: Tensor, _: Adj, *args, **kwargs) -> Tensor:
        # TODO(palowitch): Does our scaffolding ever invoke the else clause?
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(in_channels = {self.in_channels}, '
                f'hidden_channels = {self.hidden_channels}, '
                f'out_channels = {self.out_channels}, num_layers={self.num_layers}) ')


@gin.configurable
class APPNP(torch.nn.Module):
    def __init__(self, iterations: int, alpha: float,
                 in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 cached=False):
        super(APPNP, self).__init__()

        self.mlp = MLP(in_channels, hidden_channels, num_layers, out_channels, dropout, act)
        self.appnp = APPNPConv(iterations, alpha, cached=cached)

        print(self.appnp)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        x = self.mlp(x, edge_index)
        x = self.appnp(x, edge_index)
        #return F.log_softmax(x, dim=1)   # don't think we need this here
        return x


@gin.configurable
class SGC(torch.nn.Module):
    def __init__(self, iterations: int,
                 in_channels: int, hidden_channels: int,
                 out_channels: Optional[int] = None,
                 cached=False, dropout:float =0):
        super(SGC, self).__init__()

        if out_channels is None:
            out_channels = hidden_channels
        self.sgc = SGConv(in_channels=in_channels, out_channels=out_channels, K=iterations, cached=cached)
        self.dropout = dropout

        print(self.sgc)

    def reset_parameters(self):
        self.sgc.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        x = self.sgc(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #return F.log_softmax(x, dim=1) # don't think we need this here
        return x


@gin.configurable
class GATv2(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            GATv2Conv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(GATv2Conv(hidden_channels, out_channels, **kwargs))


@gin.configurable
class ARMA(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(ARMAConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                ARMAConv(hidden_channels, hidden_channels, **kwargs))


@gin.configurable
class FiLM(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(FiLMConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                FiLMConv(hidden_channels, hidden_channels, **kwargs))


@gin.configurable
class Transformer(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            TransformerConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(TransformerConv(hidden_channels, out_channels, **kwargs))


@gin.configurable
class SuperGAT(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            SuperGATConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(SuperGATConv(hidden_channels, out_channels, **kwargs))
            

@gin.configurable
class HGCN(nn.Module):
    r"""
    Hyperbolic GCN for node classification, hooked into GraphWorld.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: int,
                 c: Optional[float] = None,
                 manifold: str = 'Hyperboloid',
                 dropout: float = 0.5,
                 bias: bool = True,
                 act_name: str = 'relu',
                 **kwargs): 
        super().__init__()

        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers      = num_layers
        self.out_channels    = out_channels
        self.dropout         = dropout
        self.bias            = bias
        self.manifold_name   = manifold
        self.act             = act_name

        # 1) build a dummy 'args' namespace to feed into Chami’s code
        args = lambda **kw: None
        if manifold == "Hyperboloid":
            args.feat_dim = in_channels + 1
        else:
            args.feat_dim   = in_channels

        args.manifold   = manifold
        args.num_layers = num_layers + 1
        args.dim        = hidden_channels
        args.n_classes  = out_channels
        args.dropout    = dropout
        args.bias       = bias
        args.use_att    = False         # or True, if you want hyperbolic attention
        args.local_agg  = False         # your choice
        args.alpha      = 0.2           # only for HAT-style
        args.act        = act_name
        args.task       = 'nc'
        args.c          = c 
        args.cuda       = -1 

        if c is None:
            # learnable global curvature (init to 1.0)
            self.c = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            # fixed global curvature
            self.register_buffer('c', torch.tensor([c]))

        # 3) instantiate their encoder & decoder
        self.encoder = _HGCNEncoder(self.c, args)
        self.decoder = LinearDecoder(self.c, args)

        # for pretty-printing
        self.__repr__ = lambda: (
            f"HGCN(in={in_channels}, hid={hidden_channels}, "
            f"layers={num_layers}, out={out_channels}, manifold={manifold}, "
            f"c = {c})"
        )
        
        print(self.__repr__())

    def forward(self, x, edge_index):
        # — prep x for Hyperboloid if needed:
        if self.encoder.manifold.name == 'Hyperboloid':
            zero = torch.zeros_like(x[:, :1])
            x = torch.cat([zero, x], dim=1)

        adj = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]

        # — encode into hyperbolic embeddings
        h = self.encoder.encode(x, adj)

        # — decode to class‐scores
        logits = self.decoder.decode(h, adj)

        # — return raw logits; GraphWorld will apply loss / softmax
        return logits
    
    def reset_parameters(self):
        """
        Reset *all* trainable parameters to their initial state,
        so that each GraphWorld training run starts from fresh weights.
        """
        # 1) Reset the global curvature, if it’s a Parameter
        if isinstance(self.c, nn.Parameter):
            with torch.no_grad():
                self.c.fill_(1.0)

        # 2) Reset the Chami encoder’s own parameters
        #    It has a list of HyperbolicGraphConvolution layers, each of which
        #    has HypLinear, HypAgg, and HypAct sub‐modules.
        #    Conveniently, each of those sub‐modules implements .reset_parameters().
        for layer in self.encoder.layers:
            layer.reset_parameters()

        # 3) Reset the decoder’s linear head
        self.decoder.cls.reset_parameters()
