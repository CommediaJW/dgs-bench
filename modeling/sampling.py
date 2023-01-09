import argparse
import load_graph
import numpy
from ogb.nodeproppred import DglNodePropPredDataset
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


def bench(rank, world_size, indptr, indices, indptr_host_cache_rate,
          indices_host_cache_rate, fan_outs, seeds_num):
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
    indptr_fact_host_cache_rate = (indptr.numel() * indptr.element_size() -
                                   world_size * indptr_cache_size_per_gpu) / (
                                       indptr.numel() * indptr.element_size())
    indices_fact_host_cache_rate = (
        indices.numel() * indices.element_size() -
        world_size * indices_cache_size_per_gpu) / (indices.numel() *
                                                    indices.element_size())

    if rank == 0:
        print(
            'Seeds num {} | Indptr host cache rate {:.3f} | Indices host cache rate {:.3f}'
            .format(seeds_num, indptr_fact_host_cache_rate,
                    indices_fact_host_cache_rate))

    minimum_access_node_num_log = []
    fact_access_node_num_log = []
    access_time_log = []
    total_sampling_time_log = []
    for _ in range(10):
        seeds = torch.randint(0, graph_node_num,
                              (seeds_num, )).unique().long().cuda()
        minimum_access_node_num = 0
        fact_access_node_num = 0
        access_time = 0
        total_sampling_time = 0

        for fan_out in fan_outs:
            torch.cuda.synchronize()
            start = time.time()
            coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                seeds, chunk_indptr, chunk_indices, fan_out, False)
            torch.cuda.synchronize()
            access_time += time.time() - start
            frontier, (coo_row,
                       coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                           [seeds, coo_col], [coo_row, coo_col])
            torch.cuda.synchronize()
            total_sampling_time += time.time() - start

            minimum_access_node_num += coo_row.numel()
            for seed in seeds:
                fact_access_node_num += indptr[seed + 1] - indptr[seed]

            seeds = frontier

        minimum_access_node_num_log.append(minimum_access_node_num)
        fact_access_node_num_log.append(fact_access_node_num)
        access_time_log.append(access_time)
        total_sampling_time_log.append(total_sampling_time)

    print(
        'GPU {} | Minimum access node num {:.1f} | Fact access node num {:.1f} | Access time {:.3f} ms | Total sampling time {:.3f} ms'
        .format(rank, numpy.mean(minimum_access_node_num_log[3:]),
                numpy.mean(fact_access_node_num_log[3:]),
                numpy.mean(access_time_log[3:]) * 1000,
                numpy.mean(total_sampling_time_log[3:]) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="ogbn-papers100M",
                        choices=[
                            "reddit", "ogbn-products", "ogbn-papers100M",
                            "ogbn-papers400M", "generated"
                        ],
                        help="The dataset to be sampled.")
    parser.add_argument("--root",
                        default="/data1/",
                        help="Root path of the dataset.")
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
    del graph

    degree_log = []
    for i in range(indptr.numel() - 1):
        degree_log.append(indptr[i + 1] - indptr[i])
    print('graph = {}, average degree = {:.2f}'.format(args.dataset,
                                                       numpy.mean(degree_log)))

    for world_size in [1]:
        print("#GPU = {}".format(world_size))
        for seeds_num in [1000, 15000, 225000]:
            mp.spawn(bench,
                     args=(world_size, indptr, indices, 0, 1, [15], seeds_num),
                     nprocs=world_size)
