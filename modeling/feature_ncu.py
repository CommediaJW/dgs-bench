import torch
from dgs_create_communicator import create_dgs_communicator_single_gpu


def get_available_memory(device, num_node):
    available_mem = torch.cuda.mem_get_info(
        device)[1] - torch.ops.dgs_ops._CAPI_get_current_allocated(
        ) - torch.cuda.max_memory_allocated(
        ) - 0.3 * torch.cuda.max_memory_reserved(
        ) - 2 * 1024 * 1024 * 1024 - num_node
    available_mem = torch.tensor([available_mem]).long().cuda()
    return available_mem.cpu().numpy()[0]


def bench(rank, world_size, features, num_nodes_per_iteration,
          host_cache_rate):
    torch.manual_seed(rank)
    print('create dgs communicator')
    create_dgs_communicator_single_gpu()

    print('chunk tensor cache')
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

    print('start')

    nids = torch.randint(0, graph_num_nodes,
                         (num_nodes_per_iteration, )).unique().long().cuda()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("loading")
    _ = chunk_features._CAPI_index(nids)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("Nids len {} | Host cache rate {:.3f}".format(
        nids.numel(), host_cache_rate))


if __name__ == '__main__':
    torch.ops.load_library("../Dist-GPU-sampling/build/libdgs.so")

    torch.set_num_threads(1)
    torch.cuda.set_device(0)

    feature_dim = 128
    graph_num_nodes = 10000000
    features = torch.ones((graph_num_nodes, feature_dim)).float()
    print('finish generating features')

    bench(0, 1, features, 5000000, 0)
