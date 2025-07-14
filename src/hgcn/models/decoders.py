"""Graph decoders."""
import hgcn.manifolds as manifolds
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import hgcn.manifolds.base as Manifold

from hgcn.layers.att_layers import GraphAttentionLayer
from hgcn.layers.layers import GraphConvolution, Linear


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class HyperbolicDecoder(nn.Module):
    """
    A manifold-aware decoder for link prediction in hyperbolic space.
    Plugs directly into PyTorch Geometric's GAE model.
    """
    def __init__(self, c: torch.Tensor, manifold: Manifold):
        super().__init__()
        self.c = c
        self.manifold = manifold

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, sigmoid: bool = False) -> torch.Tensor:
        """
        Calculates a score for each edge. The GAE's recon_loss will handle
        applying the sigmoid and loss function.

        Args:
            z: Node embeddings from the encoder.
            edge_index: The edges for which to predict existence.
            sigmoid: (unused) The GAE passes this, but its loss function is
                     more stable with raw logits.

        Returns:
            A tensor of scores (logits) for each edge.
        """
        # Get the embeddings for the source and target nodes of each edge
        emb_in = z[edge_index[0]]
        emb_out = z[edge_index[1]]

        # Calculate the squared hyperbolic distance
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)

        # Return the negative distance. Smaller distance = higher score (logit).
        # This works seamlessly with BCEWithLogitsLoss used by GAE.
        return -sqdist


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}

