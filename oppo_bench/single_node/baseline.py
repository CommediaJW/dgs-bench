import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import dgl
from dgl.base import DGLError
import dgl.nn as dglnn
from dgl.utils import pin_memory_inplace
from dgl.multiprocessing import shared_tensor
import time
import numpy as np
import tqdm
from load_graph import load_papers400m_sparse, load_ogb


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
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
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

    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if num_layers > 1:
            # input projection (no residual)
            self.gat_layers.append(
                dglnn.GATConv(in_dim, num_hidden, heads[0], feat_drop,
                              attn_drop, negative_slope, False,
                              self.activation))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(
                    dglnn.GATConv(num_hidden * heads[l - 1], num_hidden,
                                  heads[l], feat_drop, attn_drop,
                                  negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(
                dglnn.GATConv(num_hidden * heads[-2], num_classes, heads[-1],
                              feat_drop, attn_drop, negative_slope, residual,
                              None))
        else:
            self.gat_layers.append(
                dglnn.GATConv(in_dim, num_classes, heads[0], feat_drop,
                              attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
        return


def train(rank, world_size, graph, model, fan_out, batch_size, bias):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)

    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    train_idx = graph.nodes()[graph.ndata['train_mask'].bool()]

    # move ids to GPU
    train_idx = train_idx.to('cuda')

    if bias:
        # bias sampling
        sampler = dgl.dataloading.NeighborSampler(
            fan_out,
            prob='probs',
            prefetch_node_feats=['features'],
            prefetch_labels=['labels'])
    else:
        # uniform sampling
        sampler = dgl.dataloading.NeighborSampler(
            fan_out,
            prefetch_node_feats=['features'],
            prefetch_labels=['labels'])

    train_dataloader = dgl.dataloading.DataLoader(graph,
                                                  train_idx,
                                                  sampler,
                                                  device='cuda',
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  use_ddp=True,
                                                  use_uva=True)

    if rank == 0:
        print('start training...')
    iteration_time_log = []
    for _ in range(1):
        model.train()

        torch.cuda.synchronize()
        start = time.time()
        for it, (input_nodes, output_nodes,
                 blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['features']
            y = blocks[-1].dstdata['labels']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            torch.cuda.synchronize()
            end = time.time()
            iteration_time_log.append(end - start)

            start = time.time()

    avg_iteration_time = np.mean(iteration_time_log[5:])
    all_gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_gather_list, avg_iteration_time)
    avg_iteration_time = np.mean(all_gather_list)
    throughput = batch_size * world_size / avg_iteration_time.item()
    if rank == 0:
        print('Time per iteration {:.3f} ms | Throughput {:.3f} seeds/sec'.
              format(avg_iteration_time * 1000, throughput))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpu',
                        default='8',
                        type=int,
                        help='The number GPU participated in the training.')
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument('--model',
                        default='graphsage',
                        choices=['graphsage', 'gat'],
                        help='The model of training.')
    parser.add_argument("--batch-size",
                        default="1000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument(
        "--dataset",
        default="ogbn-papers400M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    args = parser.parse_args()

    if args.dataset == "ogbn-products":
        graph, num_classes = load_ogb("ogbn-products", root=args.root)
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = load_ogb("ogbn-papers100M", root=args.root)
    elif args.dataset == "ogbn-papers400M":
        graph, num_classes = load_papers400m_sparse(root=args.root)

    graph = graph.formats('csc')
    graph.create_formats_()
    graph.edata.clear()

    n_procs = args.num_gpu

    fan_out = [15, 15, 15]

    if args.model == 'graphsage':
        model = SAGE(graph.ndata['features'].shape[1], 256, num_classes)
    elif args.model == 'gat':
        model = GAT(graph, 3, graph.ndata['features'].shape[1], 32,
                    num_classes, 8, F.elu, 0.6, 0.6, 0.2, False)

    if args.bias:
        graph.edata['probs'] = torch.randn((graph.num_edges(), )).float()

    print(
        'Dataset {} | GPU num {} | Model {} | Fan out {} | Batch size {} | Bias sampling {}'
        .format(args.dataset, n_procs, args.model, fan_out, args.batch_size,
                args.bias))

    import torch.multiprocessing as mp
    mp.spawn(train,
             args=(n_procs, graph, model, fan_out, args.batch_size, args.bias),
             nprocs=n_procs)
