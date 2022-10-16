# Graphsage bench

## How to run

```bash
torchrun --nproc_per_node 2 baseline.py --dataset ogbn-papers400M --batch-size 1000 --root dataset/ --print-train
```

You can use `--root` argument to specify the path of dataset.

If you add `--print-train`, the information about loss and accuracy will be printed during training.
