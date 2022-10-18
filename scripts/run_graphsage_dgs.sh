python3 graphsage/dgs_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --num-gpu 1 >> log/graphsage_dgs.log
python3 graphsage/dgs_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --num-gpu 2 >> log/graphsage_dgs.log
python3 graphsage/dgs_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --num-gpu 4 >> log/graphsage_dgs.log
python3 graphsage/dgs_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --num-gpu 8 >> log/graphsage_dgs.log
python3 graphsage/dgs_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --bias --num-gpu 1 >> log/graphsage_dgs.log
python3 graphsage/dgs_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --bias --num-gpu 2 >> log/graphsage_dgs.log
python3 graphsage/dgs_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --bias --num-gpu 4 >> log/graphsage_dgs.log
python3 graphsage/dgs_spawn.py --dataset ogbn-papers400M --batch-size 1000 --root /data1/ --bias --num-gpu 8 >> log/graphsage_dgs.log