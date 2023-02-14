from dgl.dataloading.base import BlockSampler
import torch
import dgl
import torch.distributed as dist


def create_dgs_communicator(group_size, group_rank, local_group=None):
    if group_rank == 0:
        unique_id_array = torch.ops.dgs_ops._CAPI_get_unique_id()
        broadcast_list = [unique_id_array]
    else:
        broadcast_list = [None]

    dist.broadcast_object_list(broadcast_list, 0, local_group)
    unique_ids = broadcast_list[0]
    torch.ops.dgs_ops._CAPI_set_nccl(group_size, unique_ids, group_rank)


def get_available_memory(device, model_mem_used, num_node):
    available_mem = torch.cuda.mem_get_info(device)[
        1] - model_mem_used - torch.ops.dgs_ops._CAPI_get_current_allocated(
        ) - 0.3 * torch.cuda.max_memory_reserved(
        ) - 3 * 1024 * 1024 * 1024 - num_node
    available_mem = max(available_mem, 16)
    available_mem = torch.tensor([available_mem]).long().cuda()
    dist.all_reduce(available_mem, dist.ReduceOp.MIN)
    return available_mem.cpu().numpy()[0]


class ChunkTensorSampler(BlockSampler):

    def __init__(self,
                 fanouts,
                 g,
                 cache_rate=0.0,
                 model_mem_used=0,
                 prob=None,
                 replace=False,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        torch.cuda.reset_peak_memory_stats()
        self.fanouts = fanouts
        self.prob = prob
        self.replace = replace
        self.device = torch.cuda.current_device()
        self.model_mem_used = model_mem_used

        indptr = g.adj_sparse('csc')[0]
        indices = g.adj_sparse('csc')[1]
        if self.prob:
            probs = g.edata.pop(self.prob)

        self.num_node = g.num_nodes()

        comm_size = dist.get_world_size()

        if cache_rate < 0:
            cache_rate = 0
        elif cache_rate > 1:
            cache_rate = 1

        if self.prob:
            probs_total_size = probs.numel() * probs.element_size()
            probs_cached_size_per_gpu = int(
                min(
                    probs_total_size * cache_rate // comm_size,
                    get_available_memory(self.device, self.model_mem_used,
                                         self.num_node)))
            self.chunk_probs = torch.classes.dgs_classes.ChunkTensor(
                probs.shape, probs.dtype, probs_cached_size_per_gpu)
            if dist.get_rank() == 0:
                self.chunk_probs._CAPI_load_from_tensor(probs)
                print(
                    "Cache probs per GPU {:.3f} GB, all gpu total cache rate = {:.3f}"
                    .format(
                        probs_cached_size_per_gpu / 1024 / 1024 / 1024,
                        probs_cached_size_per_gpu * comm_size /
                        probs_total_size))

        indices_total_size = indices.numel() * indices.element_size()
        indices_cached_size_per_gpu = int(
            min(
                indices_total_size * cache_rate // comm_size,
                get_available_memory(self.device, self.model_mem_used,
                                     self.num_node)))
        self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(
            indices.shape, indices.dtype, indices_cached_size_per_gpu)
        if dist.get_rank() == 0:
            self.chunk_indices._CAPI_load_from_tensor(indices)
            print(
                "Cache indices per GPU {:.3f} GB, all gpu total cache rate = {:.3f}"
                .format(
                    indices_cached_size_per_gpu / 1024 / 1024 / 1024,
                    indices_cached_size_per_gpu * comm_size /
                    indices_total_size))

        indptr_total_size = indptr.numel() * indptr.element_size()
        indptr_cached_size_per_gpu = int(
            min(
                indptr_total_size * cache_rate // comm_size,
                get_available_memory(self.device, self.model_mem_used,
                                     self.num_node)))
        self.chunk_indptr = torch.classes.dgs_classes.ChunkTensor(
            indptr.shape, indptr.dtype, indptr_cached_size_per_gpu)
        if dist.get_rank() == 0:
            self.chunk_indptr._CAPI_load_from_tensor(indptr)
            print(
                "Cache indptr per GPU {:.3f} GB, all gpu total cache rate = {:.3f}"
                .format(
                    indptr_cached_size_per_gpu / 1024 / 1024 / 1024,
                    indptr_cached_size_per_gpu * comm_size /
                    indptr_total_size))

    def __del__(self):
        del self.chunk_indices
        del self.chunk_indptr
        if (self.prob):
            del self.chunk_probs

    def get_available_mem(self):
        return get_available_memory(self.device, self.model_mem_used,
                                    self.num_node)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        seeds = seed_nodes
        blocks = []
        for fan_out in reversed(self.fanouts):
            if (self.prob):
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs_with_chunk_tensor(
                    seeds, self.chunk_indptr, self.chunk_indices,
                    self.chunk_probs, fan_out, self.replace)
            else:
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                    seeds, self.chunk_indptr, self.chunk_indices, fan_out,
                    self.replace)

            frontier, (coo_row,
                       coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                           [seeds, coo_col], [coo_row, coo_col])
            block = dgl.create_block((coo_col, coo_row),
                                     num_src_nodes=frontier.numel(),
                                     num_dst_nodes=seeds.numel())
            block.srcdata[dgl.NID] = frontier
            block.dstdata[dgl.NID] = seeds
            blocks.insert(0, block)

            seeds = frontier

        return frontier, seed_nodes, blocks
