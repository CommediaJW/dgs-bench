import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import dgl
import dgl.nn as dglnn
from dgl.base import DGLError
from dgl.utils import pin_memory_inplace
from dgl.multiprocessing import shared_tensor
import tqdm


# This function has been removed in dgl 0.9
def unpin_memory_inplace(tensor):
    """Unregister the tensor from pinned memory in-place (i.e. without copying)."""
    # needs to be writable to allow in-place modification
    try:
        dgl.backend.zerocopy_to_dgl_ndarray_for_write(tensor).unpin_memory_()
    except Exception as e:
        raise DGLError("Failed to unpin memory in-place due to: {}".format(e))


class SAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        # self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def _forward_layer(self, l, block, x):
        h = self.layers[l](block, x)
        if l != len(self.layers) - 1:
            h = F.relu(h)
            h = self.dropout(h)
        return h

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = self._forward_layer(l, blocks[l], h)
        return h

    def inference(self, g, device, batch_size):
        g.ndata['h'] = g.ndata['features']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.DataLoader(g,
                                                torch.arange(g.num_nodes(),
                                                             device=device),
                                                sampler,
                                                device=device,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=0,
                                                use_ddp=True,
                                                use_uva=True)

        for l, layer in enumerate(self.layers):
            # in order to prevent running out of GPU memory, we allocate a
            # shared output tensor 'y' in host memory, pin it to allow UVA
            # access from each GPU during forward propagation.
            y = shared_tensor((g.num_nodes(), self.n_hidden if
                               l != len(self.layers) - 1 else self.n_classes))
            pin_memory_inplace(y)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader) \
                    if dist.get_rank() == 0 else dataloader:
                x = blocks[0].srcdata['h']
                h = self._forward_layer(l, blocks[0], x)
                y[output_nodes] = h.to(y.device)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            if l > 0:
                unpin_memory_inplace(g.ndata['h'])
            if l + 1 < len(self.layers):
                # assign the output features of this layer as the new input
                # features for the next layer
                g.ndata['h'] = y
            else:
                # remove the intermediate data from the graph
                g.ndata.pop('h')
        return y


class GAT(nn.Module):

    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(in_size,
                          hid_size,
                          heads[0],
                          activation=F.elu,
                          allow_zero_in_degree=True))
        self.gat_layers.append(
            dglnn.GATConv(hid_size * heads[0],
                          hid_size,
                          heads[1],
                          residual=True,
                          activation=F.elu,
                          allow_zero_in_degree=True))
        self.gat_layers.append(
            dglnn.GATConv(hid_size * heads[1],
                          out_size,
                          heads[2],
                          residual=True,
                          activation=None,
                          allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
            h = layer(block, h)
            if l == len(self.gat_layers) - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h
