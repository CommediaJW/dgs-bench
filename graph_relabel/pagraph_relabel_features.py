import argparse
import torch
import time
import numpy
from utils.load_graph import *
from utils.dataloader import SeedGenerator
from utils.pagraph import GraphCacheServer


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

    indptr_cache_size = indptr.numel() * indptr.element_size()
    indices_cache_size = indices.numel() * indices.element_size()

    create_dgs_communicator_single_gpu()
    chunk_indptr = create_chunktensor(indptr, indptr_cache_size)
    chunk_indices = create_chunktensor(indices, indices_cache_size)

    features = graph.ndata.pop("features")
    pagraph_cacher = GraphCacheServer(features)
    if args.feat_weight:
        feat_cache_nodes_num = 1807611
        print("features cache nodes num =", feat_cache_nodes_num)
        feat_weight = torch.load(args.feat_weight)
        cache_nids = torch.argsort(
            feat_weight, descending=True)[:feat_cache_nodes_num].long().cuda()
        pagraph_cacher.cache_data(cache_nids)
    else:
        cache_nids = torch.arange(0, graph.num_nodes()).cuda()
        pagraph_cacher.cache_data(cache_nids, reorder=False)

    loading_time_log = []
    for _ in range(args.num_epochs):
        for it, seeds_nids in enumerate(train_seedloader):
            seeds = seeds_nids
            for num_picks in fan_out:
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                    seeds, chunk_indptr, chunk_indices, num_picks, False)
                frontier, (coo_row,
                           coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                               [seeds, coo_col], [coo_row, coo_col])
                seeds = frontier

            torch.cuda.synchronize()
            start = time.time()
            feature = pagraph_cacher.fetch_data(frontier)
            torch.cuda.synchronize()
            end = time.time()

            loading_time_log.append(end - start)

    print("Loading time {:.3f} ms".format(numpy.mean(loading_time_log) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="ogbn-products",
                        choices=[
                            "reddit", "ogbn-products", "ogbn-papers100M",
                            "ogbn-papers400M", "generated"
                        ])
    parser.add_argument("--root", default="/data1/")
    parser.add_argument("--batch-size",
                        default="5000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument('--fan-out', type=str, default='15,15,15')
    parser.add_argument('--libdgs',
                        default='../Dist-GPU-sampling/build/libdgs.so',
                        help='Path of libdgs.so')
    parser.add_argument('--num-epochs', type=int, default="1")
    parser.add_argument('--feat-weight', type=str, default=None)
    args = parser.parse_args()
    print(args)

    if args.dataset == "reddit":
        graph, num_classes = load_reddit()
    elif args.dataset == "ogbn-products":
        graph, num_classes = load_ogb("ogbn-products", root=args.root)
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = load_ogb("ogbn-papers100M", root=args.root)
    elif args.dataset == "ogbn-papers400M":
        graph, num_classes = load_papers400m_sparse(root=args.root,
                                                    load_true_features=False)

    run(args, graph)
