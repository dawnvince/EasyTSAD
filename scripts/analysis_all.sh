#!/bin/bash

arr=(AE AnomalyTransformer AR Donut EncDecAD FCVAE LSTMADalpha LSTMADbeta SRCNN TFAD TimesNet)
for i in ${arr[@]}
do
    python main.py --method $i --task_mode one_by_one --behavior analysis_only &
    python main.py --method $i --task_mode all_in_one --behavior analysis_only &
    python main.py --method $i --task_mode transfer_within_dataset --behavior analysis_only &
done

python main.py --method Spot --task_mode one_by_one --behavior analysis_only &
