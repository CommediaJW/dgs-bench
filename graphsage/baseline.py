import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl
import dgl.backend
from dgl.base import DGLError
import dgl.nn as dglnn
from dgl.utils import pin_memory_inplace
from dgl.multiprocessing import shared_tensor
import time
import tqdm
import numpy as np
import load_graph
import shared_graph


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


def train(rank, world_size, graph, split_idx, num_classes, batch_size, fan_out,
          print_train, dataset):
    hidden_dim = 256

    model = SAGE(graph.ndata['features'].shape[1], hidden_dim,
                 num_classes).cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    train_idx = split_idx['train_idx'].to('cuda')

    sampler = dgl.dataloading.NeighborSampler(fan_out,
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

    print("start training")

    time_log = []
    for _ in range(3):
        model.train()

        start = time.time()
        for it, (input_nodes, output_nodes,
                 blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['features']
            y = blocks[-1].dstdata['labels'].long()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 20 == 0 and rank == 0 and print_train:
                acc = MF.accuracy(y_hat, y)
                mem = torch.cuda.max_memory_allocated() / 1000000
                print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem,
                      'MB')

            torch.cuda.synchronize()
            end = time.time()
            time_log.append(end - start)

            start = time.time()

    avg_iteration_time = np.mean(time_log[1:])
    throughput = batch_size * world_size / avg_iteration_time
    if rank == 0:
        print(
            "Model GraphSAGE | Hidden dim {} | Dataset {} | Fanout {} | Batch size {} | GPU num {} | Time per iteration {:.2f} ms | Throughput {:.2f}"
            .format(hidden_dim, dataset, fan_out, batch_size, world_size,
                    avg_iteration_time * 1000, throughput))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="ogbn-papers100M",
                        choices=[
                            "reddit", "ogbn-products", "ogbn-papers100M",
                            "ogbn-papers400M"
                        ],
                        help="The dataset to be sampled.")
    parser.add_argument("--batch-size",
                        default="1000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument("--root",
                        default="dataset/",
                        help="Root path of the dataset.")
    parser.add_argument('--print-train',
                        action='store_true',
                        default=False,
                        help="Whether to print loss and acc during training.")
    args = parser.parse_args()

    torch.set_num_threads(1)
    dist.init_process_group(backend='nccl', init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    if rank == 0:
        if args.dataset == "reddit":
            graph, num_classes = load_graph.load_reddit()
        elif args.dataset == "ogbn-products":
            graph, num_classes = load_graph.load_ogb("ogbn-products",
                                                     root=args.root)
        elif args.dataset == "ogbn-papers100M":
            graph, num_classes = load_graph.load_ogb("ogbn-papers100M",
                                                     root=args.root)
        elif args.dataset == "ogbn-papers400M":
            graph, num_classes = load_graph.load_papers400m_sparse(
                root=args.root, load_true_features=False)
    else:
        graph = None
        num_classes = None

    if dist.get_world_size() > 1:
        graph, split_idx, num_classes = shared_graph.create_shared_graph(
            graph, num_classes)
    else:
        split_idx = {}
        train_mask = graph.ndata.pop('train_mask').bool()
        split_idx['train_idx'] = graph.nodes()[train_mask]
        del train_mask
        test_mask = graph.ndata.pop('test_mask').bool()
        split_idx['test_idx'] = graph.nodes()[test_mask]
        del test_mask
        val_mask = graph.ndata.pop('val_mask').bool()
        split_idx['val_idx'] = graph.nodes()[val_mask]
        del val_mask

    fan_out = [15, 15, 15]
    train(dist.get_rank(), dist.get_world_size(), graph, split_idx,
          num_classes, args.batch_size, fan_out, args.print_train,
          args.dataset)
