import torch
import numpy
import dgl
import argparse
import os
from utils.load_graph import *


def feature_relabel(feature, inversed_relabel_map):
    feature = feature.index_select(0, inversed_relabel_map)
    return feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="ogbn-products",
                        choices=[
                            "reddit", "ogbn-products", "ogbn-papers100M",
                            "ogbn-papers400M", "generated"
                        ])
    parser.add_argument("--root", default="/data1/")
    parser.add_argument("--filename")
    parser.add_argument('--libdgs',
                        default='../Dist-GPU-sampling/build/libdgs.so',
                        help='Path of libdgs.so')
    parser.add_argument('--save-root', type=str, default=None)
    args = parser.parse_args()

    tensor = torch.load(args.filename).numpy()
    inversed_relabel_map = numpy.argsort(tensor)[::-1]
    inversed_relabel_map = torch.from_numpy(inversed_relabel_map.copy())

    if args.dataset == "reddit":
        graph, num_classes = load_reddit()
    elif args.dataset == "ogbn-products":
        graph, num_classes = load_ogb("ogbn-products", root=args.root)
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = load_ogb("ogbn-papers100M", root=args.root)
    elif args.dataset == "ogbn-papers400M":
        graph, num_classes = load_papers400m_sparse(root=args.root,
                                                    load_true_features=False)

    assert (graph.num_nodes() == inversed_relabel_map.numel())
    indptr = graph.adj_sparse('csc')[0]
    indices = graph.adj_sparse('csc')[1]
    eid = graph.adj_sparse('csc')[2]

    th.ops.load_library(args.libdgs)
    relabel_graph_tensors, relabel_map = th.ops.dgs_ops._CAPI_csc_graph_relabel(
        (indptr, indices, eid), inversed_relabel_map)

    relabel_graph = dgl.graph(("csc", relabel_graph_tensors))

    for key in graph.ndata:
        relabel_graph.ndata[key] = feature_relabel(graph.ndata[key],
                                                   inversed_relabel_map)
    for key in graph.edata:
        relabel_graph.edata[key] = feature_relabel(graph.ndata[key],
                                                   relabel_graph_tensors[2])

    print("finish relabel")
    print(relabel_graph)
    if args.save_root:
        save_fn = os.path.join(args.save_root,
                               args.dataset + "_" + "relabeled.dgl")
    dgl.save_graphs(save_fn, [relabel_graph])
    print("relabeled graph saved to", save_fn)
