import argparse
import torch
import time
import numpy
from utils.load_graph import load_local_saved_graph
from utils.dataloader import SeedGenerator


def create_dgs_communicator_single_gpu():
    unique_id_array = torch.ops.dgs_ops._CAPI_get_unique_id()
    torch.ops.dgs_ops._CAPI_set_nccl(1, unique_id_array, 0)


def create_chunktensor(tensor, cache_size):
    chunk_tensor = torch.classes.dgs_classes.ChunkTensor(
        tensor.shape, tensor.dtype, cache_size)
    chunk_tensor._CAPI_load_from_tensor(tensor)

    return chunk_tensor


def run(args, graph):
    torch.ops.load_library(args.libdgs)
    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    train_nids = graph.nodes()[graph.ndata['train_mask'].bool()].cuda()
    train_seedloader = SeedGenerator(train_nids,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     drop_last=False)

    indptr = graph.adj_sparse('csc')[0]
    indices = graph.adj_sparse('csc')[1]

    cache_seeds_num = 0
    indptr_cache_size = 0 * indptr.element_size()
    indices_cache_size = 0 * indices.element_size()

    create_dgs_communicator_single_gpu()

    chunk_indptr = create_chunktensor(indptr, indptr_cache_size)
    chunk_indices = create_chunktensor(indices, indices_cache_size)

    print("Cached seeds num {}, indptr size {} B, indices size {} B".format(
        cache_seeds_num, indptr_cache_size, indices_cache_size))

    time_log = []
    hit_rate_log = []
    for _ in range(args.num_epochs):
        for it, seeds_nids in enumerate(train_seedloader):
            seeds_log = numpy.empty((0, ))
            sampling_time = 0
            seeds = seeds_nids
            for num_picks in fan_out:
                torch.cuda.synchronize()
                start = time.time()
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                    seeds, chunk_indptr, chunk_indices, num_picks, False)
                torch.cuda.synchronize()
                end = time.time()
                sampling_time += end - start

                frontier, (coo_row,
                           coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                               [seeds, coo_col], [coo_row, coo_col])

                seeds_log = numpy.union1d(seeds.cpu().numpy(), seeds_log)
                seeds = frontier

            hit_rate_log.append(
                numpy.intersect1d(seeds_log, numpy.arange(
                    0, cache_seeds_num)).size / seeds_log.size)
            time_log.append(sampling_time)

    print("Sampling time {:.3f} ms | GPU cached seeds hit rate {:.3f}".format(
        numpy.mean(time_log) * 1000, numpy.mean(hit_rate_log)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph")
    parser.add_argument("--batch-size",
                        default="5000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument('--fan-out', type=str, default='15,15,15')
    parser.add_argument('--libdgs',
                        default='../Dist-GPU-sampling/build/libdgs.so',
                        help='Path of libdgs.so')
    parser.add_argument('--num-epochs', type=int, default="1")
    args = parser.parse_args()
    print(args)

    graph, num_classes = load_local_saved_graph(args.graph)
    print(graph)

    run(args, graph)
