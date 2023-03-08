import argparse
import os
import torch
from utils.load_graph import *
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

    create_dgs_communicator_single_gpu()
    available_mem = int(
        torch.cuda.mem_get_info(0)[1] -
        torch.ops.dgs_ops._CAPI_get_current_allocated() -
        0.3 * torch.cuda.max_memory_reserved() - graph.num_nodes() -
        3 * 1024 * 1024 * 1024)
    indptr = graph.adj_sparse('csc')[0]
    chunk_indptr = create_chunktensor(
        indptr, min(available_mem,
                    indptr.numel() * indptr.element_size()))
    indices = graph.adj_sparse('csc')[1]
    chunk_indices = create_chunktensor(
        indices,
        min(
            max(
                available_mem -
                torch.ops.dgs_ops._CAPI_get_current_allocated(), 0),
            indices.numel() * indices.element_size()))

    seeds_access_count = torch.zeros((graph.num_nodes(), ))
    features_access_count = torch.zeros((graph.num_nodes(), ))

    for _ in range(args.num_epochs):
        for it, seeds_nids in enumerate(train_seedloader):
            seeds = seeds_nids
            for num_picks in fan_out:
                seeds_access_count[seeds.cpu()] += 1
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                    seeds, chunk_indptr, chunk_indices, num_picks, False)
                frontier, (coo_row,
                           coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                               [seeds, coo_col], [coo_row, coo_col])
                seeds = frontier
            features_access_count[frontier.cpu()] += 1

    seeds_access_count /= args.num_epochs
    features_access_count /= args.num_epochs

    if args.save_root:
        seeds_save_fn = os.path.join(
            args.save_root, args.dataset + "_" + str(args.num_epochs) +
            "epoch_seeds_count_info.pkl")
        features_save_fn = os.path.join(
            args.save_root, args.dataset + "_" + str(args.num_epochs) +
            "epoch_features_count_info.pkl")
        torch.save(seeds_access_count, seeds_save_fn)
        torch.save(features_access_count, features_save_fn)
        print("seeds count information saved to", seeds_save_fn)
        print("features count information saved to", features_save_fn)


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
    parser.add_argument('--save-root', type=str, default=None)
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
