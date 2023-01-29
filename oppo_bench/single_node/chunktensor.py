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
import chunktensor_sampler


def train(rank, world_size, graph, model, fan_out, batch_size, bias,
          cache_rate, libdgs):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    torch.ops.load_library(libdgs)
    chunktensor_sampler.create_dgs_communicator(world_size, rank)

    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    train_idx = graph.nodes()[graph.ndata['train_mask'].bool()]

    # move ids to GPU
    train_idx = train_idx.to('cuda')

    # cache features
    torch.cuda.reset_peak_memory_stats()
    model_mem_used = torch.cuda.max_memory_reserved()

    features = graph.ndata.pop('features')
    avaliable_mem = chunktensor_sampler.get_available_memory(
        torch.cuda.current_device(), model_mem_used, graph.num_nodes())
    features_total_size = features.numel() * features.element_size()
    features_cached_size_per_gpu = int(
        min(features_total_size * cache_rate // world_size, avaliable_mem))
    chunk_features = torch.classes.dgs_classes.ChunkTensor(
        features, features_cached_size_per_gpu)
    if dist.get_rank() == 0:
        print(
            "Cache features per GPU {:.3f} GB, all gpu total cache rate = {:.3f}"
            .format(
                features_cached_size_per_gpu / 1024 / 1024 / 1024,
                features_cached_size_per_gpu * world_size /
                features_total_size))

    # create chunktensor sampler, cache probs, indices, indptr
    if bias:
        # bias sampling
        sampler = chunktensor_sampler.ChunkTensorSampler(
            fan_out,
            graph,
            cache_rate=cache_rate,
            model_mem_used=model_mem_used,
            prob="probs",
            prefetch_labels=['labels'])
    else:
        # uniform sampling
        sampler = chunktensor_sampler.ChunkTensorSampler(
            fan_out,
            graph,
            cache_rate=cache_rate,
            model_mem_used=model_mem_used,
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
            x = chunk_features._CAPI_index(input_nodes)
            y = blocks[-1].dstdata['labels']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y.long())
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
    parser.add_argument('--libdgs',
                        default='../Dist-GPU-sampling/build/libdgs.so',
                        help='Path of libdgs.so')
    parser.add_argument(
        '--cache-rate',
        default='0.4',
        type=float,
        help=
        'The gpu cache rate of features and graph structure tensors. If the gpu memory is not enough, cache priority: features > probs > indices > indptr'
    )
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

    n_procs = min(args.num_gpu, torch.cuda.device_count())

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
             args=(n_procs, graph, model, fan_out, args.batch_size, args.bias,
                   args.cache_rate, args.libdgs),
             nprocs=n_procs)
