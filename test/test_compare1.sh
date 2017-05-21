#!/bin/bash
/usr/bin/mpirun -n 2 --bind-to none /usr/bin/python ../model_selection_new.py \
-y ./y_c.tsv \
-k ./abkt_s \
-t 1 \
-o ./out_c1
/usr/bin/mpirun -n 2 --bind-to none /usr/bin/python ../model_selection_new.py \
-y ./y_c.tsv \
-k ./abkt_s \
-x ./x_c.tsv \
-t 1 \
-o ./out_c2
/usr/bin/mpirun -n 2 --bind-to none /usr/bin/python ../model_selection_new.py \
-y ./y_c.tsv \
-k ./abkt_s \
-x ./x_c.tsv \
-t 1 \
-o ./out_c3
/usr/bin/mpirun -n 2 --bind-to none /usr/bin/python ../model_selection_new.py \
-y ./y_c.tsv \
-k ./abkt_s \
-x ./x_c.tsv \
-t 1 \
-o ./out_c4
