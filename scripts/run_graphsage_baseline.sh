python3 graphsage/baseline_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --num-gpu 1 >> log/graphsage_baseline.log
python3 graphsage/baseline_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --num-gpu 2 >> log/graphsage_baseline.log
python3 graphsage/baseline_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --num-gpu 4 >> log/graphsage_baseline.log
python3 graphsage/baseline_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --num-gpu 8 >> log/graphsage_baseline.log
python3 graphsage/baseline_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --bias --num-gpu 1 >> log/graphsage_baseline.log
python3 graphsage/baseline_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --bias --num-gpu 2 >> log/graphsage_baseline.log
python3 graphsage/baseline_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --bias --num-gpu 4 >> log/graphsage_baseline.log
python3 graphsage/baseline_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --bias --num-gpu 8 >> log/graphsage_baseline.log