import torch
import time
import torch.distributed as dist
from GraphCache.cache import get_node_value, get_cache_nids

PRINT = False


def print_memory():
    if PRINT:
        print("{:.2f}, {:.2f}".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024))


def preprocess_for_cached_nids_out_degrees(graph, available_mem, devices_num, device_id):
    start = time.time()

    available_mem = torch.tensor([available_mem]).long().cuda()
    dist.all_reduce(available_mem, dist.ReduceOp.MIN)
    available_mem = available_mem.cpu().numpy()[0]

    avg_feature_size = graph["features"].shape[1] * graph[
        "features"].element_size()

    cache_nids_num = min(available_mem // avg_feature_size, (graph["features"].shape[0] + devices_num - 1) // devices_num)

    if dist.get_rank() == 0:
        cached_nids = torch.argsort(graph["out_degrees"], descending=True)[:cache_nids_num  * devices_num]
        cached_nids = cached_nids[torch.randperm(cached_nids.shape[0])]
    else:
        cached_nids = None
    broadcast_list = [cached_nids]
    dist.broadcast_object_list(broadcast_list, 0)
    cached_nids = broadcast_list[0][device_id * cache_nids_num: (device_id + 1) * cache_nids_num]

    end = time.time()
    print("GPU {}, preprocess time for cache nids (by out_degrees): {:.3f} seconds".format(
        device_id, end - start))

    return cached_nids


def preprocess_for_cached_nids_heat(graph, sampling_heat, feature_heat, bias, available_mem, devices_num, device_id):
    start = time.time()

    available_mem = torch.tensor([available_mem]).long().cuda()
    dist.all_reduce(available_mem, dist.ReduceOp.MIN)
    available_mem = available_mem.cpu().numpy()[0]

    sampling_heat = sampling_heat.cuda()
    feature_heat = feature_heat.cuda()
    dist.reduce(sampling_heat, 0, dist.ReduceOp.SUM)
    dist.reduce(feature_heat, 0, dist.ReduceOp.SUM)
    sampling_heat /= devices_num
    feature_heat /= devices_num

    if dist.get_rank() == 0:

        avg_feature_size = graph["features"].shape[1] * graph[
            "features"].element_size()
        indptr = graph["indptr"]
        indices = graph["indices"]
        num_nodes = indptr.shape[0] - 1
        avg_degree = indptr[-1].item() / num_nodes
        in_degrees = indptr[1:] - indptr[0:num_nodes]

        if bias:
            avg_structure_size = indptr.element_size() + avg_degree * (
                indices.element_size() + graph["probs"].element_size())
        else:
            avg_structure_size = indptr.element_size() + avg_degree * (
                indices.element_size())

        sampling_nids, sampling_cache_bytes, sampling_value, feature_nids, feature_cache_bytes, feature_value = get_node_value(
            sampling_heat, feature_heat,
            1, 1, in_degrees.cuda(), indptr.element_size(), 
            avg_structure_size, avg_feature_size, bias)

        structure_cache_nids, feature_cache_nids = get_cache_nids(
            sampling_nids, sampling_cache_bytes, sampling_value, feature_nids,
            feature_cache_bytes, feature_value, available_mem)

        structure_cache_nids_per_gpu = structure_cache_nids.shape[0] // devices_num
        feature_cache_nids_per_gpu = feature_cache_nids.shape[0] // devices_num

        structure_cache_nids = structure_cache_nids[:structure_cache_nids_per_gpu * devices_num].cpu()
        feature_cache_nids = feature_cache_nids[:feature_cache_nids_per_gpu * devices_num].cpu()

        structure_cache_nids = structure_cache_nids[torch.randperm(structure_cache_nids.shape[0])]
        feature_cache_nids = feature_cache_nids[torch.randperm(feature_cache_nids.shape[0])]

    else:
        structure_cache_nids_per_gpu = None
        feature_cache_nids_per_gpu = None
        structure_cache_nids = None
        feature_cache_nids = None

    broadcast_list = [structure_cache_nids_per_gpu, feature_cache_nids_per_gpu, structure_cache_nids, feature_cache_nids]
    dist.broadcast_object_list(broadcast_list, 0)

    structure_cache_nids = broadcast_list[2][device_id * broadcast_list[0]:(device_id + 1) * broadcast_list[0]]
    feature_cache_nids = broadcast_list[3][device_id * broadcast_list[1]:(device_id + 1) * broadcast_list[1]]

    end = time.time()
    print("GPU {}, preprocess time for cache nids: {:.3f} seconds".format(
        device_id, end - start))

    return structure_cache_nids, feature_cache_nids
