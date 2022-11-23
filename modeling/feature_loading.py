import argparse
import numpy
import time
import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator


def get_available_memory(device, num_node):
    available_mem = torch.cuda.mem_get_info(
        device)[1] - torch.ops.dgs_ops._CAPI_get_current_allocated(
        ) - 0.3 * torch.cuda.max_memory_reserved(
        ) - 2 * 1024 * 1024 * 1024 - num_node
    available_mem = max(available_mem, 16)
    available_mem = torch.tensor([available_mem]).long().cuda()
    dist.all_reduce(available_mem, dist.ReduceOp.MIN)
    return available_mem.cpu().numpy()[0]


def get_hit_rate(cached_nids, nids):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    nids = nids.cpu().numpy()

    cached_num_nodes_per_gpu = cached_nids.shape[0] // world_size
    local_cached_nids = cached_nids[cached_num_nodes_per_gpu *
                                    rank:cached_num_nodes_per_gpu * (rank + 1)]

    local_hit_num = numpy.intersect1d(nids, local_cached_nids).shape[0]
    remote_hit_num = numpy.intersect1d(nids,
                                       cached_nids).shape[0] - local_hit_num

    total_num = nids.shape[0]
    local_hit_rate = local_hit_num / total_num
    remote_hit_rate = remote_hit_num / total_num
    return local_hit_rate, remote_hit_rate


def bench(rank, world_size, libdgs, features, num_nodes_per_iteration):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    torch.ops.load_library(libdgs)
    create_dgs_communicator(world_size, rank)

    if rank == 0:
        print("Chunktensor cahce...")
    avaliable_mem = get_available_memory(
        rank, features.shape[0]) - 1 * 1024 * 1024 * 1024
    feature_cache_size_per_gpu = min(
        avaliable_mem,
        features.numel() * features.element_size() // world_size)
    cached_num_nodes = feature_cache_size_per_gpu // (features.shape[1] *
                                                      features.element_size())
    feature_cache_size_per_gpu = cached_num_nodes * features.shape[
        1] * features.element_size()
    cached_nids = torch.arange(0, cached_num_nodes).long().numpy()
    cache_rate_per_gpu = feature_cache_size_per_gpu / (features.numel() *
                                                       features.element_size())
    chunk_features = torch.classes.dgs_classes.ChunkTensor(
        features, feature_cache_size_per_gpu)

    graph_num_nodes = features.shape[0]

    if rank == 0:
        print("Start...")
    time_log = []
    local_hit_rate_log = []
    remote_hit_rate_log = []
    for it in range(50):
        nids = torch.randint(0, graph_num_nodes,
                             (num_nodes_per_iteration, )).unique().long()
        hit_rate = get_hit_rate(cached_nids, nids)
        local_hit_rate_log.append(hit_rate[0])
        remote_hit_rate_log.append(hit_rate[1])

        nids = nids.cuda()
        torch.cuda.synchronize()
        start = time.time()
        _ = chunk_features._CAPI_index(nids)
        torch.cuda.synchronize()
        time_log.append(time.time() - start)

    print(
        "GPU {} | Local cache rate {:.3f} | Loading time {:.2f} ms | Loacl hit rate {:.3f} | Remote hit rate {:.3f}"
        .format(rank, cache_rate_per_gpu,
                numpy.mean(time_log[5:]) * 1000,
                numpy.mean(local_hit_rate_log[5:]),
                numpy.mean(remote_hit_rate_log[5:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--libdgs",
                        default="../Dist-GPU-sampling/build/libdgs.so",
                        help="Path of libdgs.so")
    args = parser.parse_args()

    torch.manual_seed(1)

    feature_dim = 128
    graph_num_nodes = 111059956
    num_nodes_per_iteration = 208000
    print("Generate feature data...")
    features = torch.ones((graph_num_nodes, feature_dim)).float()

    import torch.multiprocessing as mp
    for world_size in [2]:
        mp.spawn(bench,
                 args=(world_size, args.libdgs, features,
                       num_nodes_per_iteration),
                 nprocs=world_size)
