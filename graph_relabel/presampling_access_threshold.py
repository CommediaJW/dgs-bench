import torch
import numpy
import argparse
from utils.load_graph import *

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
    parser.add_argument("--threshold", type=float, default="1.0")
    parser.add_argument("--sampling-info", action="store_true", default=False)
    parser.add_argument("--loading-info", action="store_true", default=False)
    args = parser.parse_args()

    tensor = torch.load(args.filename).numpy()
    total_access_times = numpy.sum(tensor)
    print("Analysis result for", args.filename)
    print("Sum of access times: {}".format(total_access_times))

    if args.threshold >= 1:
        accessed_nodes = numpy.nonzero(tensor)[0]
        accessed_nodes_num = accessed_nodes.size
    else:
        total_access_times = numpy.sum(tensor)
        sorted_tensor = numpy.sort(tensor)[::-1]
        sum_access_times = 0
        accessed_nodes_num = 0
        for elem in sorted_tensor:
            sum_access_times += elem.item()
            accessed_nodes_num += 1
            if sum_access_times >= total_access_times * args.threshold:
                break
        accessed_nodes = numpy.sort(
            numpy.argsort(tensor)[::-1][:accessed_nodes_num])

    print("To cover {:.1f}% access times, minimum nodes num = {}".format(
        args.threshold * 100, accessed_nodes_num))

    if args.loading_info or args.sampling_info:
        if args.dataset == "reddit":
            graph, num_classes = load_reddit()
        elif args.dataset == "ogbn-products":
            graph, num_classes = load_ogb("ogbn-products", root=args.root)
        elif args.dataset == "ogbn-papers100M":
            graph, num_classes = load_ogb("ogbn-papers100M", root=args.root)
        elif args.dataset == "ogbn-papers400M":
            graph, num_classes = load_papers400m_sparse(
                root=args.root, load_true_features=False)

        assert (graph.num_nodes() == tensor.size)

        if args.loading_info:
            print("Feature dim =", graph.ndata["features"].shape[1])
            print("Feature type size =",
                  graph.ndata["features"].element_size())

        if args.sampling_info:
            indptr = graph.adj_sparse('csc')[0]
            indptr_accessed_elem_num = numpy.union1d(accessed_nodes,
                                                     accessed_nodes + 1).size
            indices_accessed_elem_num = 0
            for nid in accessed_nodes:
                indices_accessed_elem_num += indptr[nid.item() +
                                                    1] - indptr[nid.item()]

            print("Indptr minimum accessed element num = {}".format(
                indptr_accessed_elem_num))
            print("Indices minimum accessed element num = {}".format(
                indices_accessed_elem_num))
