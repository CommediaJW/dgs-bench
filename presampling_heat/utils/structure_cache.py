import torch
import torch.distributed as dist
import time
import dgl


class StructureCacheServer:

    def __init__(self, indptr, indices, probs=None, device_id=0):
        self.indptr = indptr
        self.indices = indices
        self.probs = probs

        self.cached_indptr = None
        self.cached_indices = None
        self.cached_probs = None

        self.device_id = device_id

        self.cached_nids_hashed = None
        self.cached_nids_in_gpu_hashed = None

        self.full_cached = False
        self.no_cache = False

        self.access_times = 0
        self.hit_times = 0

    def cache_data(self, cache_nids, full_cached=False):
        self.full_cached = full_cached

        start = time.time()

        if self.full_cached:

            self.cached_indptr = self.indptr.cuda(self.device_id)
            self.cached_indices = self.indices.cuda(self.device_id)
            if self.probs is not None:
                self.cached_probs = self.probs.cuda(self.device_id)

            indptr_cached_size = self.indptr.element_size(
            ) * self.indptr.numel()
            indices_cached_size = self.indices.element_size(
            ) * self.indices.numel()
            if self.probs is not None:
                probs_cached_size = self.probs.element_size(
                ) * self.probs.numel()

        else:
            if cache_nids.numel() > 0:
                cache_nids = cache_nids.cuda(self.device_id)
                self.cached_nids_hashed, self.cached_nids_in_gpu_hashed = torch.ops.dgs_ops._CAPI_create_cache_hashmap(
                    cache_nids)

                self.cached_indptr = torch.ops.dgs_ops._CAPI_get_sub_indptr(
                    cache_nids, self.indptr).cuda(self.device_id)

                self.cached_indices = torch.ops.dgs_ops._CAPI_get_sub_edge_data(
                    cache_nids, self.indptr, self.cached_indptr,
                    self.indices).cuda(self.device_id)

                if self.probs is not None:
                    self.cached_probs = torch.ops.dgs_ops._CAPI_get_sub_edge_data(
                        cache_nids, self.indptr, self.cached_indptr,
                        self.probs).cuda(self.device_id)

                indptr_cached_size = self.cached_indptr.element_size(
                ) * self.cached_indptr.numel()
                indices_cached_size = self.cached_indices.element_size(
                ) * self.cached_indices.numel()
                if self.probs is not None:
                    probs_cached_size = self.cached_probs.element_size(
                    ) * self.cached_probs.numel()
            else:
                self.no_cache = True
                indptr_cached_size = 0
                indices_cached_size = 0
                if self.probs is not None:
                    probs_cached_size = 0

        end = time.time()

        print("GPU {} takes {:.3f} s to cache structure data".format(
            self.device_id, end - start))
        print(
            "GPU {} Indptr cache size = {:.3f} GB, cache rate = {:.3f}".format(
                self.device_id, indptr_cached_size / 1024 / 1024 / 1024,
                indptr_cached_size /
                (self.indptr.element_size() * self.indptr.numel())))
        print("GPU {} Indices cache size = {:.3f} GB, cache rate = {:.3f}".
              format(
                  self.device_id, indices_cached_size / 1024 / 1024 / 1024,
                  indices_cached_size /
                  (self.indices.element_size() * self.indices.numel())))
        if self.probs is not None:
            print("GPU {} Probs cache size = {:.3f} GB, cache rate = {:.3f}".
                  format(
                      self.device_id, probs_cached_size / 1024 / 1024 / 1024,
                      probs_cached_size /
                      (self.probs.element_size() * self.probs.numel())))

    def sample_neighbors(self,
                         seeds_nids,
                         fan_out,
                         replace=False,
                         count=False,
                         sampling_heat=None,
                         log_hit_rate=False):
        seeds = seeds_nids.cuda(self.device_id)
        blocks = []

        for num_picks in reversed(fan_out):

            if count:
                sampling_heat[seeds] += 1

            if self.full_cached:
                if log_hit_rate:
                    self.access_times += seeds.numel()
                    self.hit_times += seeds.numel()
                if self.probs is not None:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_bias(
                        seeds, self.cached_indptr, self.cached_indices,
                        self.cached_probs, num_picks, replace)
                else:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors(
                        seeds, self.cached_indptr, self.cached_indices,
                        num_picks, replace)

            elif self.no_cache:
                if log_hit_rate:
                    self.access_times += seeds.numel()
                if self.probs is not None:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_bias(
                        seeds, self.indptr, self.indices, self.probs,
                        num_picks, replace)
                else:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors(
                        seeds, self.indptr, self.indices, num_picks, replace)

            else:
                if log_hit_rate:
                    local_seeds_num = torch.ops.dgs_ops._CAPI_count_local_cache_nids(
                        seeds, self.cached_nids_hashed,
                        self.cached_nids_in_gpu_hashed)
                    self.access_times += seeds.numel()
                    self.hit_times += local_seeds_num
                if self.probs is not None:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_bias_with_caching(
                        seeds, self.cached_indptr, self.indptr,
                        self.cached_indices, self.indices, self.cached_probs,
                        self.probs, self.cached_nids_hashed,
                        self.cached_nids_in_gpu_hashed, num_picks, replace)
                else:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_caching(
                        seeds, self.cached_indptr, self.indptr,
                        self.cached_indices, self.indices,
                        self.cached_nids_hashed,
                        self.cached_nids_in_gpu_hashed, num_picks, replace)

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

        return frontier, seeds_nids, blocks

    def print_hit_rate(self):
        if self.access_times > 0:
            print("GPU {}, structure cache hit rate = {:.3f}".format(
                self.device_id, self.hit_times / self.access_times))
        else:
            print("GPU {} didn't log any hit information!".format(
                self.device_id))

    def clear_cache(self):
        del self.cached_indptr
        del self.cached_indices
        if self.probs is not None:
            del self.cached_probs
        del self.cached_nids_in_gpu_hashed
        del self.cached_nids_hashed

        self.cached_indptr = None
        self.cached_indices = None
        self.cached_probs = None
        self.cached_nids_hashed = None
        self.cached_nids_in_gpu_hashed = None
        self.full_cached = False
        self.no_cache = False
        self.access_times = 0
        self.hit_times = 0


class StructureP2PCacheServer:

    def __init__(self, indptr, indices, probs=None, process_group=None):
        self.indptr = indptr
        self.indices = indices
        self.probs = probs
        self.cached_indptr = None
        self.cached_indices = None
        self.cached_probs = None

        self.process_group = process_group
        self.device_id = dist.get_rank(self.process_group)
        self.num_devices = dist.get_world_size(self.process_group)

        self.cached_nids_hashed = None
        self.cached_nids_in_gpu_hashed = None
        self.device_idx_hashed = None

        self.full_cached = False
        self.no_cache = False

        self.access_times = 0
        self.local_hit_times = 0
        self.remote_hit_times = 0

    def cache_data(self,
                   cache_nids,
                   total_cache_nids_num=None,
                   full_cached=False):
        self.full_cached = full_cached

        start = time.time()

        if self.full_cached:

            self.cached_indptr = self.indptr.cuda(self.device_id)
            self.cached_indices = self.indices.cuda(self.device_id)
            if self.probs is not None:
                self.cached_probs = self.probs.cuda(self.device_id)
            indptr_cached_size = self.indptr.element_size(
            ) * self.indptr.numel()
            indices_cached_size = self.indices.element_size(
            ) * self.indices.numel()
            if self.probs is not None:
                probs_cached_size = self.probs.element_size(
                ) * self.probs.numel()

        else:
            if cache_nids.numel() > 0:

                cache_nids_cuda = cache_nids.cuda(self.device_id)

                sub_indptr = torch.ops.dgs_ops._CAPI_get_sub_indptr(
                    cache_nids_cuda, self.indptr).cuda(self.device_id)
                torch.cuda.empty_cache()
                self.cached_indptr = torch.classes.dgs_classes.TensorP2PServer(
                    sub_indptr.shape, sub_indptr.dtype)
                self.cached_indptr._CAPI_load_device_tensor_data(sub_indptr)
                indptr_cached_size = sub_indptr.element_size(
                ) * sub_indptr.numel()

                sub_indices = torch.ops.dgs_ops._CAPI_get_sub_edge_data(
                    cache_nids_cuda, self.indptr, sub_indptr,
                    self.indices).cuda(self.device_id)
                torch.cuda.empty_cache()
                self.cached_indices = torch.classes.dgs_classes.TensorP2PServer(
                    sub_indices.shape, sub_indices.dtype)
                self.cached_indices._CAPI_load_device_tensor_data(sub_indices)
                indices_cached_size = sub_indices.element_size(
                ) * sub_indices.numel()
                del sub_indices

                if self.probs is not None:
                    sub_probs = torch.ops.dgs_ops._CAPI_get_sub_edge_data(
                        cache_nids_cuda, self.indptr, sub_indptr,
                        self.probs).cuda(self.device_id)
                    torch.cuda.empty_cache()
                    self.cached_probs = torch.classes.dgs_classes.TensorP2PServer(
                        sub_probs.shape, sub_probs.dtype)
                    self.cached_probs._CAPI_load_device_tensor_data(sub_probs)
                    probs_cached_size = sub_probs.element_size(
                    ) * sub_probs.numel()
                    del sub_probs

                del sub_indptr, cache_nids_cuda

                cache_nids_list = [None for _ in range(self.num_devices)]
                cache_nids_list[self.device_id] = cache_nids.cpu()
                dist.all_gather_object(cache_nids_list,
                                       cache_nids_list[self.device_id],
                                       self.process_group)

                if total_cache_nids_num is None:
                    total_cache_nids_num = torch.cat(
                        cache_nids_list).unique().numel()
                for i in range(self.num_devices):
                    torch.ops.dgs_ops._CAPI_tensor_pin_memory(
                        cache_nids_list[i])

                self.cached_nids_hashed, self.cached_nids_in_gpu_hashed, self.device_idx_hashed = torch.ops.dgs_ops._CAPI_create_p2p_cache_hashmap(
                    cache_nids_list, total_cache_nids_num, self.device_id)

                for i in range(self.num_devices):
                    torch.ops.dgs_ops._CAPI_tensor_unpin_memory(
                        cache_nids_list[i])

            else:
                self.no_cache = True
                indptr_cached_size = 0
                indices_cached_size = 0
                if self.probs is not None:
                    probs_cached_size = 0

        end = time.time()

        print("GPU {} takes {:.3f} s to cache structure data".format(
            self.device_id, end - start))
        print(
            "GPU {} Indptr cache size = {:.3f} GB, cache rate = {:.3f}".format(
                self.device_id, indptr_cached_size / 1024 / 1024 / 1024,
                indptr_cached_size /
                (self.indptr.element_size() * self.indptr.numel())))
        print("GPU {} Indices cache size = {:.3f} GB, cache rate = {:.3f}".
              format(
                  self.device_id, indices_cached_size / 1024 / 1024 / 1024,
                  indices_cached_size /
                  (self.indices.element_size() * self.indices.numel())))
        if self.probs is not None:
            print("GPU {} Probs cache size = {:.3f} GB, cache rate = {:.3f}".
                  format(
                      self.device_id, probs_cached_size / 1024 / 1024 / 1024,
                      probs_cached_size /
                      (self.probs.element_size() * self.probs.numel())))

    def sample_neighbors(self,
                         seeds_nids,
                         fan_out,
                         replace=False,
                         count=False,
                         sampling_heat=None,
                         log_hit_rate=False):
        seeds = seeds_nids.cuda(self.device_id)
        blocks = []

        for num_picks in reversed(fan_out):

            if count:
                sampling_heat[seeds] += 1

            if self.full_cached:
                if log_hit_rate:
                    self.access_times += seeds.numel()
                    self.local_hit_times += seeds.numel()
                if self.probs is not None:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_bias(
                        seeds, self.cached_indptr, self.cached_indices,
                        self.cached_probs, num_picks, replace)
                else:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors(
                        seeds, self.cached_indptr, self.cached_indices,
                        num_picks, replace)

            elif self.no_cache:
                if log_hit_rate:
                    self.access_times += seeds.numel()
                if self.probs is not None:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_bias(
                        seeds, self.indptr, self.indices, self.probs,
                        num_picks, replace)
                else:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors(
                        seeds, self.indptr, self.indices, num_picks, replace)

            else:
                if log_hit_rate:
                    local_seeds_num, remote_seeds_num = torch.ops.dgs_ops._CAPI_count_local_and_remote_p2p_cache_nids(
                        seeds, self.cached_nids_hashed, self.device_idx_hashed,
                        self.device_id)
                    self.access_times += seeds.numel()
                    self.local_hit_times += local_seeds_num
                    self.remote_hit_times += remote_seeds_num
                if self.probs is not None:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_bias_with_p2p_caching(
                        seeds, self.cached_indptr, self.indptr,
                        self.cached_indices, self.indices, self.cached_probs,
                        self.probs, self.cached_nids_hashed,
                        self.cached_nids_in_gpu_hashed, self.device_idx_hashed,
                        num_picks, replace)
                else:
                    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_p2p_caching(
                        seeds, self.cached_indptr, self.indptr,
                        self.cached_indices, self.indices,
                        self.cached_nids_hashed,
                        self.cached_nids_in_gpu_hashed, self.device_idx_hashed,
                        num_picks, replace)

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

        return frontier, seeds_nids, blocks

    def print_hit_rate(self):
        if self.access_times > 0:
            print("GPU {}, structure p2p cache local hit rate = {:.3f}".format(
                self.device_id, self.local_hit_times / self.access_times))
            print(
                "GPU {}, structure p2p cache remote hit rate = {:.3f}".format(
                    self.device_id, self.remote_hit_times / self.access_times))
        else:
            print("GPU {} didn't log any hit information!".format(
                self.device_id))

    def clear_cache(self):
        del self.cached_indptr
        del self.cached_indices
        if self.probs is not None:
            del self.cached_probs
        del self.cached_nids_hashed
        del self.cached_nids_in_gpu_hashed
        del self.device_idx_hashed

        self.cached_indptr = None
        self.cached_indices = None
        self.cached_probs = None
        self.cached_nids_hashed = None
        self.cached_nids_in_gpu_hashed = None
        self.device_idx_hashed = None
        self.full_cached = False
        self.no_cache = False
        self.access_times = 0
        self.local_hit_times = 0
        self.remote_hit_times = 0
