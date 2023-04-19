from dgl.dataloading.base import BlockSampler
import torch
import dgl
import torch.distributed as dist


def create_dgs_communicator(group_size, local_group=None):
    if dist.get_rank() % group_size == 0:
        unique_id_array = torch.ops.dgs_ops._CAPI_get_unique_id()
        broadcast_list = [unique_id_array]
    else:
        broadcast_list = [None]

    dist.broadcast_object_list(broadcast_list,
                               dist.get_rank() - dist.get_rank() % group_size,
                               local_group)
    unique_ids = broadcast_list[0]
    torch.ops.dgs_ops._CAPI_set_nccl(group_size, unique_ids,
                                     dist.get_rank() % group_size)


def create_chunktensor(tensor,
                       num_gpus,
                       available_mem,
                       cache_rate=1.0,
                       root_rank=0,
                       local_group=None):

    if dist.get_rank() == root_rank:
        total_size = tensor.numel() * tensor.element_size()
        cached_size_per_gpu = int(
            min(total_size * cache_rate // num_gpus, available_mem))
        broadcast_list = [tensor.shape, tensor.dtype, cached_size_per_gpu]
    else:
        broadcast_list = [None, None, None]

    dist.broadcast_object_list(broadcast_list, root_rank, local_group)

    chunk_tensor = torch.classes.dgs_classes.ChunkTensor(
        broadcast_list[0], broadcast_list[1], broadcast_list[2])

    if dist.get_rank() == root_rank:
        chunk_tensor._CAPI_load_from_tensor(tensor)
        print(
            "Cache size per GPU {:.3f} GB, all gpus total cache rate = {:.3f}".
            format(cached_size_per_gpu / 1024 / 1024 / 1024,
                   cached_size_per_gpu * num_gpus / total_size))

    dist.barrier()
    return chunk_tensor


class ChunkTensorSampler:

    def __init__(self,
                 fanouts,
                 chunk_indptr,
                 chunk_indices,
                 chunk_probs=None,
                 replace=False):
        torch.cuda.reset_peak_memory_stats()
        self.chunk_indptr = chunk_indptr
        self.chunk_indices = chunk_indices
        self.chunk_probs = chunk_probs
        self.fanouts = fanouts
        self.replace = replace
        self.device = torch.cuda.current_device()

    def __del__(self):
        del self.chunk_indices
        del self.chunk_indptr
        if self.chunk_probs is not None:
            del self.chunk_probs

    def sample_blocks(self, seed_nodes):
        seeds = seed_nodes.cuda()
        blocks = []
        for fan_out in reversed(self.fanouts):
            if self.chunk_probs is not None:
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
