import argparse
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl
from models import DistSAGE


def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    arr.uniform_(-1, 1)
    return arr


def run(args, device, data):
    # Unpack data
    train_nid, num_classes, in_feats, g = data

    # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet
    sampler = dgl.dataloading.NeighborSampler([15, 15, 15])
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = DistSAGE(in_feats, 256, num_classes)
    model = model.to(device)

    dev_id = g.rank() % args.num_gpus
    model = th.nn.parallel.DistributedDataParallel(model,
                                                   device_ids=[dev_id],
                                                   output_device=dev_id)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # Training loop
    iteration_time_log = []
    for epoch in range(args.num_epochs):

        th.cuda.synchronize()
        start = time.time()
        with model.join():
            for it, (input_nodes, seeds, blocks) in enumerate(dataloader):
                batch_inputs = g.ndata["features"][input_nodes].to(device)
                batch_labels = g.ndata["labels"][seeds].to(device)
                blocks = [block.to(device) for block in blocks]
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                th.cuda.synchronize()
                end = time.time()
                iteration_time_log.append(end - start)

                start = time.time()

    avg_iteration_time = np.mean(iteration_time_log[5:])
    print("Part {} | Iteration Time {:.4f} ms | Throughput {:.3f} seeds/sec".
          format(g.rank(), avg_iteration_time * 1000,
                 args.batch_size / avg_iteration_time))


def main(args):
    dgl.distributed.initialize(args.ip_config)
    th.distributed.init_process_group(backend="gloo")
    g = dgl.distributed.DistGraph(args.graph_name,
                                  part_config=args.part_config)
    print("Rank {}".format(g.rank()))

    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata["train_mask"],
                                           pb,
                                           force_even=True)
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print("Part {} | train nid num {} (local num {})".format(
        g.rank(), len(train_nid),
        len(np.intersect1d(train_nid.numpy(), local_nid))))
    dev_id = g.rank() % args.num_gpus
    device = th.device("cuda:" + str(dev_id))
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    num_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print("#Labels {}".format(num_classes))
    in_feats = g.ndata["features"].shape[1]
    print("Feature dim {}".format(in_feats))

    # Pack data and run
    data = train_nid, num_classes, in_feats, g
    run(args, device, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument("--ip-config",
                        type=str,
                        help="The file for IP configuration")
    parser.add_argument("--num-gpu",
                        type=int,
                        default=1,
                        help="the number of GPU device.")
    parser.add_argument("--part-config",
                        type=str,
                        help="The path to the partition config file")
    parser.add_argument("--num-classes",
                        type=int,
                        help="the number of classes")
    parser.add_argument("--model",
                        default="graphsage",
                        choices=["graphsage", "gat"],
                        help="The model of training.")
    parser.add_argument("--bias",
                        action="store_true",
                        default=False,
                        help="Sample with bias.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=1)
    args = parser.parse_args()

    print(args)
    main(args)
