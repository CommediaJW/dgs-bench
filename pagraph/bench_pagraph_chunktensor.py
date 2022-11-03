import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import dgl
import dgl.backend
import time
import numpy as np
import load_graph
from chunk_tensor_sampler import ChunkTensorSampler
from pagraph import GraphCacheServer
from graphsage import SAGE
from dgs_create_communicator import create_dgs_communicator


def train(rank, world_size, graph, num_classes, batch_size, fan_out, dataset,
          bias, libdgs):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    torch.ops.load_library(libdgs)
    create_dgs_communicator(world_size, rank)

    feat = graph.ndata.pop('features')

    if rank == 0:
        print("create model")
    hidden_dim = 256
    model = SAGE(feat.shape[1], hidden_dim, num_classes).cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    torch.cuda.reset_peak_memory_stats()
    model_mem_used = torch.cuda.max_memory_reserved()
    if rank == 0:
        print("create sampler")
    if bias:
        sampler = ChunkTensorSampler(fan_out,
                                     graph,
                                     model_mem_used=model_mem_used,
                                     prob="prob",
                                     cache_rate=0,
                                     prefetch_labels=['labels'])
    else:
        sampler = ChunkTensorSampler(fan_out,
                                     graph,
                                     model_mem_used=model_mem_used,
                                     cache_rate=0,
                                     prefetch_labels=['labels'])

    if rank == 0:
        print("pagraph cache")
    avaliable_mem = sampler.get_available_mem() - 1 * 1024 * 1024 * 1024
    cacher = GraphCacheServer(feat, rank)
    cacher.cache_data_chunk_tensor(
        cacher.get_cache_nid(graph, avaliable_mem * dist.get_world_size()))

    if rank == 0:
        print("create dataloader")
    train_idx = graph.nodes()[graph.ndata['train_mask'].bool()].to('cuda')
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
        print("start training")
    iteration_time_log = []
    sample_time_log = []
    load_time_log = []
    train_time_log = []
    for _ in range(1):
        model.train()

        iteration_start = time.time()

        for it, (input_nodes, output_nodes,
                 blocks) in enumerate(train_dataloader):
            torch.cuda.synchronize()
            sample_time_log.append(time.time() - iteration_start)

            load_start = time.time()
            x = cacher.fetch_data_chunk_tensor(input_nodes)
            y = blocks[-1].dstdata['labels'].long()
            torch.cuda.synchronize()
            load_time_log.append(time.time() - load_start)

            train_start = time.time()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            train_time_log.append(time.time() - train_start)

            torch.cuda.synchronize()
            iteration_time_log.append(time.time() - iteration_start)

            if it > 100:
                break

            iteration_start = time.time()

    avg_iteration_time = np.mean(iteration_time_log[10:])
    avg_sample_time = np.mean(sample_time_log[10:]) * 1000
    avg_load_time = np.mean(load_time_log[10:]) * 1000
    avg_train_time = np.mean(train_time_log[10:]) * 1000
    throughput = batch_size * world_size / avg_iteration_time

    if rank == 0:
        if bias:
            print(
                "Model GraphSAGE | Sample with bias | Hidden dim {} | Dataset {} | Fanout {} | Batch size {} | GPU num {}"
                .format(hidden_dim, dataset, fan_out, batch_size, world_size))
        else:
            print(
                "Model GraphSAGE | Sample without bias | Hidden dim {} | Dataset {} | Fanout {} | Batch size {} | GPU num {}"
                .format(hidden_dim, dataset, fan_out, batch_size, world_size))

    torch.cuda.synchronize()
    print(
        "GPU {} | Iteration time {:.2f} ms | Sample time {:.2f} ms | Load time {:.2f} ms | Train time {:.2f} ms | Throughput {:.2f}"
        .format(dist.get_rank(), avg_iteration_time * 1000, avg_sample_time,
                avg_load_time, avg_train_time, throughput))


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
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument("--libdgs",
                        default="../Dist-GPU-sampling/build/libdgs.so",
                        help="Path of libdgs.so")
    args = parser.parse_args()

    torch.manual_seed(1)

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

    print("create csc formats")
    graph = graph.formats('csc')
    graph.create_formats_()
    graph.edata.clear()
    print("finish load graph")

    indptr = graph.adj_sparse('csc')[0]
    indices = graph.adj_sparse('csc')[1]
    feat = graph.ndata['features']

    print(
        "Dataset {}, indptr size {:.2f} MB, indices size {:.2f} MB, feature size {:.2f} MB"
        .format(args.dataset,
                indptr.numel() * indptr.element_size() / 1024 / 1024,
                indices.numel() * indices.element_size() / 1024 / 1024,
                feat.numel() * feat.element_size() / 1024 / 1024))

    if args.bias:
        print("generate probs tensor")
        probs = torch.rand(graph.num_edges()).float()
        graph.edata['prob'] = probs

        print("probs size {:.2f} MB".format(
            probs.numel() * probs.element_size() / 1024 / 1024))

    fan_out = [15, 15, 15]

    n_procs_set = [2, 1]
    import torch.multiprocessing as mp
    for n_procs in n_procs_set:
        mp.spawn(train,
                 args=(n_procs, graph, num_classes, args.batch_size, fan_out,
                       args.dataset, args.bias, args.libdgs),
                 nprocs=n_procs)
