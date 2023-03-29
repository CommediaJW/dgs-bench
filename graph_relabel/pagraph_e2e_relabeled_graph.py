import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy
from utils.load_graph import *
from utils.dataloader import SeedGenerator
from utils.pagraph import GraphCacheServer
from utils.models import SAGE
from utils.chunktensor_sampler import ChunkTensorSampler


def create_dgs_communicator_single_gpu():
    unique_id_array = torch.ops.dgs_ops._CAPI_get_unique_id()
    torch.ops.dgs_ops._CAPI_set_nccl(1, unique_id_array, 0)


def create_chunktensor(tensor, cache_size):
    chunk_tensor = torch.classes.dgs_classes.ChunkTensor(
        tensor.shape, tensor.dtype, cache_size)
    chunk_tensor._CAPI_load_from_tensor(tensor)

    return chunk_tensor


def run(args, graph, num_classes):
    torch.ops.load_library(args.libdgs)
    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    train_nids = graph.nodes()[graph.ndata['train_mask'].bool()].cuda()
    train_seedloader = SeedGenerator(train_nids,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     drop_last=False)

    # sampling_weight = torch.load(args.sampling_weight)
    # sampling_relabel_map = torch.argsort(sampling_weight,
    #                                      descending=True).long()
    # feat_weight = torch.load(args.feat_weight)
    # feat_weight = feat_weight.index_select(0, sampling_relabel_map)

    features = graph.ndata.pop("features")
    # pagraph_cacher = GraphCacheServer(features)
    # feat_cache_nodes_num = 1
    # print("features cache nodes num =", feat_cache_nodes_num)
    # if args.feat_weight:
    #     cache_nids = torch.argsort(
    #         feat_weight, descending=True)[:feat_cache_nodes_num].long().cuda()
    #     pagraph_cacher.cache_data(cache_nids, True)
    # else:
    #     cache_nids, reorder = pagraph_cacher.get_cache_nid(
    #         graph,
    #         feat_cache_nodes_num * features.element_size() * features.shape[1])
    #     pagraph_cacher.cache_data(cache_nids, reorder)

    indptr = graph.adj_sparse('csc')[0]
    indices = graph.adj_sparse('csc')[1]
    create_dgs_communicator_single_gpu()
    feature_cache_size = 0
    chunk_feature = create_chunktensor(features, feature_cache_size)
    print("feature cache size =", feature_cache_size / 1024 / 1024, "MB")
    indptr_cache_size = 0
    chunk_indptr = create_chunktensor(indptr, indptr_cache_size)
    print("indptr cache size =", indptr_cache_size / 1024 / 1024, "MB")
    indices_cache_size = 0
    chunk_indices = create_chunktensor(indices, indices_cache_size)
    print("indices cache size =", indices_cache_size / 1024 / 1024, "MB")

    sampler = ChunkTensorSampler(fan_out, chunk_indptr, chunk_indices)
    model = SAGE(features.shape[1], 256, num_classes).cuda()
    loss_fcn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    sampling_time_log = []
    loading_time_log = []
    training_time_log = []
    iteration_time_log = []
    for _ in range(args.num_epochs):
        for it, seeds_nids in enumerate(train_seedloader):
            torch.cuda.synchronize()
            total_start = start = time.time()
            frontier, seeds, blocks = sampler.sample_blocks(seeds_nids)
            torch.cuda.synchronize()
            end = time.time()
            sampling_time_log.append(end - start)

            torch.cuda.synchronize()
            start = time.time()
            x = chunk_feature._CAPI_index(frontier)
            torch.cuda.synchronize()
            end = time.time()
            loading_time_log.append(end - start)

            y = graph.ndata["labels"].index_select(
                0, seeds.cpu()).flatten().long().cuda()
            torch.cuda.synchronize()
            start = time.time()
            pred = model(blocks, x)
            loss = loss_fcn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            training_time_log.append(end - start)
            iteration_time_log.append(end - total_start)

    print(
        "Sampling time {:.3f} ms | Loading time {:.3f} ms | Training time {:.3f} ms | Iteration time {:.3f} ms"
        .format(
            numpy.mean(sampling_time_log[3:]) * 1000,
            numpy.mean(loading_time_log[3:]) * 1000,
            numpy.mean(training_time_log[3:]) * 1000,
            numpy.mean(iteration_time_log[3:]) * 1000))


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
    parser.add_argument('--sampling-weight', type=str, default=None)
    parser.add_argument('--feat-weight', type=str, default=None)
    args = parser.parse_args()
    print(args)

    graph, num_classes = load_local_saved_graph(args.graph)
    print(graph)

    run(args, graph, num_classes)
