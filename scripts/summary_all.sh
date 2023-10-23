#!/bin/bash

# summary_item=(csv_summary cdf_summary dis_each_curve)
summary_item=csv_summary
for i in ${summary_item[@]}
do
    python main.py --task_mode one_by_one --behavior $i &
    python main.py --task_mode all_in_one --behavior $i &
    python main.py --task_mode transfer_within_dataset --behavior $i
done

