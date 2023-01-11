import argparse
import load_graph
import numpy
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dgs_create_communicator import create_dgs_communicator

torch.ops.load_library('../Dist-GPU-sampling/build/libdgs.so')


def get_available_memory(device, num_node):
    available_mem = torch.cuda.mem_get_info(
        device)[1] - torch.ops.dgs_ops._CAPI_get_current_allocated(
        ) - torch.cuda.max_memory_allocated(
        ) - 0.3 * torch.cuda.max_memory_reserved(
        ) - 2 * 1024 * 1024 * 1024 - num_node
    available_mem = torch.tensor([available_mem]).long().cuda()
    dist.all_reduce(available_mem, dist.ReduceOp.MIN)
    return available_mem.cpu().numpy()[0]


def bench(rank, world_size, indptr, indices, probs, indptr_host_cache_rate,
          indices_host_cache_rate, probs_host_cache_rate, fan_outs, seeds_num):
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    create_dgs_communicator(world_size, rank)

    graph_node_num = indptr.numel() - 1

    avaliable_men = get_available_memory(rank, graph_node_num)
    indptr_cache_size_per_gpu = int(
        min(avaliable_men, (1 - indptr_host_cache_rate) * indptr.numel() *
            indptr.element_size() // world_size))
    chunk_indptr = torch.classes.dgs_classes.ChunkTensor(
        indptr, indptr_cache_size_per_gpu)
    avaliable_men = get_available_memory(rank, graph_node_num)
    indices_cache_size_per_gpu = int(
        min(avaliable_men, (1 - indices_host_cache_rate) * indices.numel() *
            indices.element_size() // world_size))
    chunk_indices = torch.classes.dgs_classes.ChunkTensor(
        indices, indices_cache_size_per_gpu)
    if probs is not None:
        avaliable_men = get_available_memory(rank, graph_node_num)
        probs_cache_size_per_gpu = int(
            min(avaliable_men, (1 - probs_host_cache_rate) * probs.numel() *
                probs.element_size() // world_size))
        chunk_probs = torch.classes.dgs_classes.ChunkTensor(
            probs, probs_cache_size_per_gpu)
        probs_fact_host_cache_rate = (
            probs.numel() * probs.element_size() - world_size *
            probs_cache_size_per_gpu) / (probs.numel() * probs.element_size())
    indptr_fact_host_cache_rate = (indptr.numel() * indptr.element_size() -
                                   world_size * indptr_cache_size_per_gpu) / (
                                       indptr.numel() * indptr.element_size())
    indices_fact_host_cache_rate = (
        indices.numel() * indices.element_size() -
        world_size * indices_cache_size_per_gpu) / (indices.numel() *
                                                    indices.element_size())

    minimum_access_node_num_log = []
    sampling_time_log = []
    fact_seeds_num_log = []
    for _ in range(10):
        seeds = torch.randint(0, graph_node_num,
                              (seeds_num, )).unique().long().cuda()
        fact_seeds_num = 0
        minimum_access_node_num = 0
        sampling_time = 0
        for fan_out in fan_outs:
            if probs is not None:
                torch.cuda.synchronize()
                start = time.time()
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs_with_chunk_tensor(
                    seeds, chunk_indptr, chunk_indices, chunk_probs, fan_out,
                    False)
                torch.cuda.synchronize()
                sampling_time += time.time() - start
            else:
                torch.cuda.synchronize()
                start = time.time()
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                    seeds, chunk_indptr, chunk_indices, fan_out, False)
                torch.cuda.synchronize()
                sampling_time += time.time() - start
            frontier, (coo_row,
                       coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                           [seeds, coo_col], [coo_row, coo_col])

            fact_seeds_num += seeds.numel()
            minimum_access_node_num += coo_row.numel()

            seeds = frontier

        fact_seeds_num_log.append(fact_seeds_num)
        minimum_access_node_num_log.append(minimum_access_node_num)
        sampling_time_log.append(sampling_time)

    if probs is not None:
        print(
            'GPU {} | #Seeds {:.2f} | Indptr host cache rate {:.3f} | Indices host cache rate {:.3f} | Probs host cache rate {:.3f} | #Sampled node {:.1f} | Sampling time {:.3f} ms'
            .format(rank, numpy.mean(fact_seeds_num_log[3:]),
                    indptr_fact_host_cache_rate, indices_fact_host_cache_rate,
                    probs_fact_host_cache_rate,
                    numpy.mean(minimum_access_node_num_log[3:]),
                    numpy.mean(sampling_time_log[3:]) * 1000))
    else:
        print(
            'GPU {} | #Seeds {:.2f} | Indptr host cache rate {:.3f} | Indices host cache rate {:.3f} | #Sampled node {:.1f} | Sampling time {:.3f} ms'
            .format(rank, numpy.mean(fact_seeds_num_log[3:]),
                    indptr_fact_host_cache_rate, indices_fact_host_cache_rate,
                    numpy.mean(minimum_access_node_num_log[3:]),
                    numpy.mean(sampling_time_log[3:]) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="ogbn-products",
                        choices=[
                            "reddit", "ogbn-products", "ogbn-papers100M",
                            "ogbn-papers400M", "generated"
                        ])
    parser.add_argument("--root", default="/data1/")
    parser.add_argument('--bias', action='store_true', default=False)
    args = parser.parse_args()

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
    elif args.dataset == "generated":
        graph, num_classes = load_graph.load_rand_generated()

    indptr = graph.adj_sparse("csc")[0].long()
    indices = graph.adj_sparse("csc")[1].long()
    if args.bias:
        print('sampling with bias')
        probs = torch.rand(graph.num_edges()).float()
    else:
        print('sampling without bias')
        probs = None
    del graph

    degree_log = []
    for i in range(indptr.numel() - 1):
        degree_log.append(indptr[i + 1] - indptr[i])
    print('graph = {}, average degree = {:.2f}'.format(args.dataset,
                                                       numpy.mean(degree_log)))

    for world_size in [1]:
        print("#GPU = {}".format(world_size))
        for seeds_num in [300000, 400000, 500000, 750000, 1000000]:
            mp.spawn(bench,
                     args=(world_size, indptr, indices, probs, 0, 0, 0, [15],
                           seeds_num),
                     nprocs=world_size)
