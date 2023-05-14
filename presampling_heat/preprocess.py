import torch
import time
import torch.distributed as dist
from GraphCache.cache import get_node_value, get_cache_nids


def preprocess_for_cached_nids_out_degrees(features, out_degrees,
                                           available_mem, device_id):
    start = time.time()

    avg_feature_size = features.shape[1] * features.element_size()

    cache_nids_num = min(available_mem // avg_feature_size, features.shape[0])

    cached_nids = torch.argsort(out_degrees, descending=True)[:cache_nids_num]

    end = time.time()
    print(
        "GPU {}, preprocess time for cache nids (by out_degrees): {:.3f} seconds"
        .format(device_id, end - start))

    return cached_nids


def get_node_value(sampling_heat,
                   feature_heat,
                   in_degrees,
                   graph_type_size,
                   avg_feature_bytes,
                   with_probs=False):
    sampling_nids = torch.nonzero(sampling_heat).squeeze()
    feature_nids = torch.nonzero(feature_heat).squeeze()

    sampling_cache_bytes = None
    if with_probs:
        sampling_cache_bytes = (graph_type_size +
                                in_degrees[sampling_nids].int() *
                                (graph_type_size + 4))  # 4 bytes for float

    else:
        sampling_cache_bytes = (graph_type_size +
                                in_degrees[sampling_nids].int() *
                                (graph_type_size))

    feature_cache_bytes = torch.full_like(feature_nids,
                                          avg_feature_bytes,
                                          dtype=torch.int32)

    sampling_value = sampling_heat[sampling_nids] / sampling_cache_bytes
    feature_value = feature_heat[feature_nids] / avg_feature_bytes

    return sampling_nids, sampling_cache_bytes, sampling_value, feature_nids, feature_cache_bytes, feature_value


def preprocess_for_cached_nids_heat(graph, sampling_heat, feature_heat, bias,
                                    available_mem, device_id):
    start = time.time()

    avg_feature_size = graph["features"].shape[1] * graph[
        "features"].element_size()
    indptr = graph["indptr"]
    num_nodes = indptr.shape[0] - 1
    in_degrees = indptr[1:] - indptr[0:num_nodes]

    sampling_nids, sampling_cache_bytes, sampling_value, feature_nids, feature_cache_bytes, feature_value = get_node_value(
        sampling_heat, feature_heat, in_degrees.cuda(), indptr.element_size(),
        avg_feature_size, bias)

    structure_cache_nids, feature_cache_nids = get_cache_nids(
        sampling_nids, sampling_cache_bytes, sampling_value, feature_nids,
        feature_cache_bytes, feature_value, available_mem)

    end = time.time()
    print("GPU {}, preprocess time for cache nids: {:.3f} seconds".format(
        device_id, end - start))

    return structure_cache_nids, feature_cache_nids
