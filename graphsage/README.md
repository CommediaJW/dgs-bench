# Graphsage bench

## How to run

```bash
python3 graphsage/baseline_spawn.py --dataset ogbn-papers100M --root /data1/ --print-train --bias --num-gpu 4
python3 graphsage/dgs_spawn.py --dataset ogbn-papers100M --root /data1/ --libdgs PATH_TO_LIBDGS --print-train --bias --num-gpu 4
```

You can use `--root` argument to specify the path of dataset.

If you add `--print-train`, the information about loss and accuracy will be printed during training.

If you add `--bias`, the sampler will sample neighbors with probability.

`--libdgs` specify the path to `libdgs.so`, for example, `../Dist-GPU-sampling/build/libdgs.so`
