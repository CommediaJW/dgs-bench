import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import dgl
import time
import numpy as np
from load_graph import load_papers400m_sparse, load_ogb
from models import SAGE, GAT


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
        heads = [8, 8, 8]
        model = GAT(graph.ndata['features'].shape[1], 32, num_classes, heads)

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
