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


def bench(indptr, indices, probs, cache_rates, seeds_num, fan_outs):
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    create_dgs_communicator_single_gpu()

    graph_node_num = indptr.numel() - 1
    available_mem = get_available_memory(0, graph_node_num)

    indptr_cache_size = min(
        available_mem,
        int(indptr.numel() * indptr.element_size() * cache_rates[0]))
    chunk_indptr = torch.torch.classes.dgs_classes.ChunkTensor(
        indptr.shape, indptr.dtype, indptr_cache_size)
    chunk_indptr._CAPI_load_from_tensor(indptr)

    indices_cache_size = min(
        available_mem - torch.ops.dgs_ops._CAPI_get_current_allocated(),
        int(indices.numel() * indices.element_size() * cache_rates[1]))
    chunk_indices = torch.torch.classes.dgs_classes.ChunkTensor(
        indices.shape, indices.dtype, indices_cache_size)
    chunk_indices._CAPI_load_from_tensor(indices)

    probs_cache_size = min(
        available_mem - torch.ops.dgs_ops._CAPI_get_current_allocated(),
        int(probs.numel() * probs.element_size() * cache_rates[2]))
    chunk_probs = torch.torch.classes.dgs_classes.ChunkTensor(
        probs.shape, probs.dtype, probs_cache_size)
    chunk_probs._CAPI_load_from_tensor(probs)

    seeds = torch.randint(0, graph_node_num,
                          (seeds_num, )).unique().long().cuda()
    fact_seeds_num = 0
    minimum_access_node_num = 0
    for fan_out in fan_outs:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("sampling")
        coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs_with_chunk_tensor(
            seeds, chunk_indptr, chunk_indices, chunk_probs, fan_out, False)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        frontier, (coo_row, coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
            [seeds, coo_col], [coo_row, coo_col])

        fact_seeds_num += seeds.numel()
        minimum_access_node_num += coo_row.numel()

        seeds = frontier

    print(
        '#Seeds {:.2f} | [indptr, indices, probs] cache rates {} | #Sampled node {:.1f}'
        .format(fact_seeds_num, cache_rates, minimum_access_node_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=10000000)
    parser.add_argument("--degree", type=int, default=15)
    args = parser.parse_args()
    print(args)

    indptr = torch.arange(0, args.num_nodes + 1).long() * args.degree
    indices = torch.randint(0, args.num_nodes,
                            (args.num_nodes * args.degree, ))
    num_edges = args.num_nodes * args.degree
    probs = torch.randn((num_edges, )).abs().float()

    bench(indptr, indices, probs, [1, 1, 1], 5000, [10])
