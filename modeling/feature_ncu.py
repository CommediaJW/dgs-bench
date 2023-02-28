import argparse
import torch
from dgs_create_communicator import create_dgs_communicator_single_gpu

torch.ops.load_library('../Dist-GPU-sampling/build/libdgs.so')


def get_available_memory(device, num_node):
    available_mem = torch.cuda.mem_get_info(
        device)[1] - torch.ops.dgs_ops._CAPI_get_current_allocated(
        ) - torch.cuda.max_memory_allocated(
        ) - 0.3 * torch.cuda.max_memory_reserved(
        ) - 2 * 1024 * 1024 * 1024 - num_node
    available_mem = torch.tensor([available_mem]).long().cuda()
    return available_mem.cpu().numpy()[0]


def bench(features, cache_rate, num_nids_per_iteration):
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    create_dgs_communicator_single_gpu()

    avaliable_mem = get_available_memory(
        0, features.shape[0]) - 2 * 1024 * 1024 * 1024
    features_cache_size = min(
        avaliable_mem,
        cache_rate * features.numel() * features.element_size())
    chunk_features = torch.torch.classes.dgs_classes.ChunkTensor(
        features.shape, features.dtype, features_cache_size)
    chunk_features._CAPI_load_from_tensor(features)

    graph_num_nodes = features.shape[0]

    nids = torch.randint(0, graph_num_nodes,
                         (num_nids_per_iteration, )).unique().long().cuda()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("loading")
    _ = chunk_features._CAPI_index(nids)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("Nids len {} | Cache rate {:.3f}".format(nids.numel(), cache_rate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=10000000)
    parser.add_argument("--feat-dim", type=int, default=128)
    args = parser.parse_args()
    print(args)

    features = torch.ones((args.num_nodes, args.feat_dim)).float()

    bench(features, 0, 5000000)
