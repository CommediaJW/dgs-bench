import argparse
import os
import time
import torch
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
    parser.add_argument('--fan-out', type=str, default='15,15,15')
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

    start = time.time()

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    train_nids = graph.nodes()[graph.ndata['train_mask'].bool()]
    reversed_graph = dgl.reverse(graph, copy_ndata=False)
    del graph

    reversed_graph.ndata["_P"] = torch.zeros(reversed_graph.num_nodes())
    reversed_graph.ndata["_P"][train_nids] = 1
    for layer in range(len(fan_out)):
        reversed_graph.ndata['_p'] = torch.minimum(
            reversed_graph.ndata['_P'].mul(
                fan_out[len(fan_out) - layer - 1]).div(
                    reversed_graph.out_degrees().to(torch.float32)),
            torch.ones(reversed_graph.num_nodes()))
        reversed_graph.update_all(dgl.function.copy_u("_p", "m"),
                                  dgl.function.sum("m", "_tp"))
        sampling_weight = reversed_graph.ndata["_P"]
        reversed_graph.ndata["_P"] = reversed_graph.ndata["_P"].add(
            reversed_graph.ndata["_tp"])

    feature_weight = reversed_graph.ndata.pop("_P")

    end = time.time()
    print("Presampling time {:.3f} s".format(end - start))

    if args.save_root:
        feature_save_fn = os.path.join(
            args.save_root, args.dataset + "_presampling_" + args.fan_out +
            "_feature_weight.pkl")
        torch.save(feature_weight, feature_save_fn)
        print("feature weight saved to", feature_save_fn)
        sampling_save_fn = os.path.join(
            args.save_root, args.dataset + "_presampling_" + args.fan_out +
            "_sampling_weight.pkl")
        torch.save(sampling_weight, sampling_save_fn)
        print("sampling weight saved to", sampling_save_fn)
