import argparse
import numpy
import time
import torch

torch.ops.load_library('../Dist-GPU-sampling/build/libdgs.so')


def create_dgs_communicator_single_gpu():
    unique_id_array = torch.ops.dgs_ops._CAPI_get_unique_id()
    torch.ops.dgs_ops._CAPI_set_nccl(1, unique_id_array, 0)


def create_chunktensor(tensor, name, cache_size):
    chunk_tensor = torch.classes.dgs_classes.ChunkTensor(
        tensor.shape, tensor.dtype, cache_size)

    chunk_tensor._CAPI_load_from_tensor(tensor)
    print("{} cache size {:.3f} GB, cache rate = {:.3f}".format(
        name, cache_size / 1024 / 1024 / 1024,
        cache_size / (tensor.numel() * tensor.element_size())))

    return chunk_tensor


def cache_data(data, available_mem):
    indptr, indices, probs, features = data

    indptr_size = indptr.numel() * indptr.element_size()
    indices_size = indices.numel() * indices.element_size()
    probs_size = probs.numel() * probs.element_size()
    features_size = features.numel() * features.element_size()

    print("Indptr size {:.3f} GB".format(indptr_size / 1024 / 1024 / 1024))
    print("Indices size {:.3f} GB".format(indices_size / 1024 / 1024 / 1024))
    print("Probs size {:.3f} GB".format(probs_size / 1024 / 1024 / 1024))
    print("Features size {:.3f} GB".format(features_size / 1024 / 1024 / 1024))

    create_dgs_communicator_single_gpu()

    # indptr_cache_size = 0
    indptr_cache_size = min(
        indptr_size,
        available_mem - torch.torch.ops.dgs_ops._CAPI_get_current_allocated())
    chunk_indptr = create_chunktensor(indptr, "indptr", indptr_cache_size)

    indices_cache_size = 0
    # indices_cache_size = min(
    #     indices_size,
    #     available_mem - torch.torch.ops.dgs_ops._CAPI_get_current_allocated())
    chunk_indices = create_chunktensor(indices, "indices", indices_cache_size)

    # probs_cache_size = 0
    probs_cache_size = min(
        probs_size,
        available_mem - torch.torch.ops.dgs_ops._CAPI_get_current_allocated())
    chunk_probs = create_chunktensor(probs, "probs", probs_cache_size)

    features_cache_size = 0
    # features_cache_size = min(
    #     features_size,
    #     available_mem - torch.torch.ops.dgs_ops._CAPI_get_current_allocated())
    chunk_features = create_chunktensor(features, "features",
                                        features_cache_size)

    return chunk_indptr, chunk_indices, chunk_probs, chunk_features


def run(args, data):
    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    available_men = 4 * 1024 * 1024 * 1024
    chunk_indptr, chunk_indices, chunk_probs, chunk_features = cache_data(
        data, available_men)

    sampling_time_log = []
    loading_time_log = []
    total_time_log = []

    for i in range(30):
        seeds = torch.randint(0, args.num_nodes,
                              (args.batch_size, )).unique().long().cuda()
        torch.cuda.synchronize()
        total_start = time.time()
        for num_picks in fan_out:
            coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs_with_chunk_tensor(
                seeds, chunk_indptr, chunk_indices, chunk_probs, num_picks,
                False)
            frontier, (coo_row,
                       coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                           [seeds, coo_col], [coo_row, coo_col])
            seeds = frontier
        torch.cuda.synchronize()
        end = time.time()
        sampling_time_log.append(end - total_start)

        del seeds, coo_row, coo_col

        start = time.time()
        _ = chunk_features._CAPI_index(frontier)
        torch.cuda.synchronize()
        end = time.time()
        loading_time_log.append(end - start)
        total_time_log.append(end - total_start)

    print(
        "Sampling time {:.3f} ms | Loading time {:.3f} ms | Total time {:.3f} ms"
        .format(
            numpy.mean(sampling_time_log[3:]) * 1000,
            numpy.mean(loading_time_log[3:]) * 1000,
            numpy.mean(total_time_log[3:]) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=10000000)
    parser.add_argument("--degree", type=int, default=15)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--batch-size",
                        default="5000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument('--fan-out', type=str, default='15,15,15')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)

    indptr = torch.arange(0, args.num_nodes + 1).long() * args.degree
    indices = torch.randint(0, args.num_nodes,
                            (args.num_nodes * args.degree, ))
    num_edges = args.num_nodes * args.degree
    probs = torch.randn((num_edges, )).abs().float()
    features = torch.ones((args.num_nodes, args.feat_dim)).float()

    data = indptr, indices, probs, features
    run(args, data)
