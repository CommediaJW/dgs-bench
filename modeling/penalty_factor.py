import argparse
import numpy
import time
import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator


def get_available_memory(device, num_node):
    available_mem = torch.cuda.mem_get_info(
        device)[1] - torch.ops.dgs_ops._CAPI_get_current_allocated(
        ) - torch.cuda.max_memory_allocated(
        ) - 0.3 * torch.cuda.max_memory_reserved(
        ) - 2 * 1024 * 1024 * 1024 - num_node
    available_mem = torch.tensor([available_mem]).long().cuda()
    dist.all_reduce(available_mem, dist.ReduceOp.MIN)
    return available_mem.cpu().numpy()[0]


def bench(rank, world_size, libdgs, features, num_nodes_per_iteration,
          host_cache_rate, bandwidth_local, bandwidth_inter_gpu,
          bandwidth_pcie):
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    torch.ops.load_library(libdgs)
    create_dgs_communicator(world_size, rank)

    avaliable_mem = get_available_memory(
        rank, features.shape[0]) - 1 * 1024 * 1024 * 1024
    feature_cache_size_per_gpu = min(avaliable_mem,
                                     (1 - host_cache_rate) * features.numel() *
                                     features.element_size() // world_size)
    cached_num_nodes_per_gpu = int(
        feature_cache_size_per_gpu //
        (features.shape[1] * features.element_size()))
    feature_cache_size_per_gpu = cached_num_nodes_per_gpu * features.shape[
        1] * features.element_size()
    chunk_features = torch.classes.dgs_classes.ChunkTensor(
        features, feature_cache_size_per_gpu)

    graph_num_nodes = features.shape[0]

    local_time_log = []
    local_len_log = []
    remote_time_log = []
    remote_len_log = []
    host_time_log = []
    host_len_log = []
    nids_len_log = []
    for it in range(10):
        nids = torch.randint(
            0, graph_num_nodes,
            (num_nodes_per_iteration, )).unique().long().cuda()
        nids_len_log.append(nids.numel())
        local_nids, remote_nids, host_nids = chunk_features._CAPI_split_index(
            nids)
        del nids
        local_len = local_nids.numel()
        local_len_log.append(local_len)
        remote_len = remote_nids.numel()
        remote_len_log.append(remote_len)
        host_len = host_nids.numel()
        host_len_log.append(host_len)
        torch.cuda.synchronize()
        start = time.time()
        if local_len > 0:
            _ = chunk_features._CAPI_local_index(local_nids)
        torch.cuda.synchronize()
        local_time_log.append(time.time() - start)
        start = time.time()
        if remote_len > 0:
            _ = chunk_features._CAPI_remote_index(remote_nids)
        torch.cuda.synchronize()
        remote_time_log.append(time.time() - start)
        start = time.time()
        if host_len > 0:
            _ = chunk_features._CAPI_host_index(host_nids)
        torch.cuda.synchronize()
        host_time_log.append(time.time() - start)

    total_len = numpy.mean(nids_len_log[3:])
    local_len = numpy.mean(local_len_log[3:])
    remote_len = numpy.mean(remote_len_log[3:])
    host_len = numpy.mean(host_len_log[3:])

    local_time = numpy.mean(local_time_log[3:]) * 1000
    remote_time = numpy.mean(remote_time_log[3:]) * 1000
    host_time = numpy.mean(host_time_log[3:]) * 1000

    local_infer = features.shape[1] * features.element_size(
    ) * local_len / 1024 / 1024 / 1024 / bandwidth_local * 1000
    remote_infer = features.shape[1] * features.element_size(
    ) * remote_len / 1024 / 1024 / 1024 / bandwidth_inter_gpu * 1000
    host_infer = features.shape[1] * features.element_size(
    ) * host_len / 1024 / 1024 / 1024 / bandwidth_pcie * 1000

    print(
        "GPU {} | Total nids len {:.1f} | host cahce rate {:.3f} | local time infer {:.2f}, fact {:.2f}, factor {:.2f} | remote time infer {:.2f}, fact {:.2f}, factor {:.2f} | host time infer {:.2f}, fact {:.2f}, factor {:.2f}"
        .format(rank, total_len, host_cache_rate, local_infer, local_time,
                local_time / local_infer, remote_infer, remote_time,
                remote_time / remote_infer, host_infer, host_time,
                host_time / host_infer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--libdgs",
                        default="../Dist-GPU-sampling/build/libdgs.so",
                        help="Path of libdgs.so")
    args = parser.parse_args()

    feature_dim = 128
    graph_num_nodes = 20000000

    # unit: GB/s
    bandwidth_local = 1300
    bandwidth_inter_gpu = 262.65
    bandwidth_pcie = 31.05

    import torch.multiprocessing as mp
    for world_size in [2]:
        features = torch.ones(
            (graph_num_nodes * world_size, feature_dim)).float()
        print("#GPU = {}".format(world_size))
        for host_cache_rate in [0.2, 0.4, 0.6]:
            for nodes_per_iteration in [
                    5000000, 7500000, 10000000, 15000000, 20000000
            ]:
                mp.spawn(bench,
                         args=(world_size, args.libdgs, features,
                               nodes_per_iteration, host_cache_rate,
                               bandwidth_local, bandwidth_inter_gpu,
                               bandwidth_pcie),
                         nprocs=world_size)
