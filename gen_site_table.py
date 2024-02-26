import pandas as pd
import numpy as np

mode = ["one_by_one", "all_in_one", "zero_shot"]
dataset_delay = [
    ("AIOPS", 5),
    ("NAB", 8),
    ("TODS", 4),
    ("WSD", 5),
    ("Yahoo", 4),
    ("UCR", 7),
    ("NEK", 5)
]
metric_map = {
    "best f1 under pa": "point-wise F1",
    "event-based f1 under pa with mode log": "reduced-length F1",
    "event-based f1 under pa with mode squeeze": "event-wise F1",
    "event-based f1 under 3-delay pa with mode log": "delay reduced-length F1", # 4
    "event-based f1 under 10-delay pa with mode log": "delay reduced-length F1", # 5
    "event-based f1 under 20-delay pa with mode log": "delay reduced-length F1", # 6
    "event-based f1 under 50-delay pa with mode log": "delay reduced-length F1", # 7
    "event-based f1 under 150-delay pa with mode log": "delay reduced-length F1", # 8
    
    "point-based auprc pa": "point-wise AUPRC",
    "event-based auprc under pa with mode log": "reduced-length AUPRC",
    "event-based auprc under pa with mode squeeze": "event-wise AUPRC", 
    "3-th auprc under event-based pa with mode log": "delay reduced-length AUPRC",
    "10-th auprc under event-based pa with mode log": "delay reduced-length AUPRC",
    "20-th auprc under event-based pa with mode log": "delay reduced-length AUPRC",
    "50-th auprc under event-based pa with mode log": "delay reduced-length AUPRC",
    "150-th auprc under event-based pa with mode log": "delay reduced-length AUPRC"
}

all_res = {}
all_res_data = {}
for mod in mode:
    file_path = "Results/Summary/CSVs/{}.csv".format(mod)
    
    if mod == "one_by_one":
        mod = "naive"

    old_df = pd.read_csv(file_path,  header=None)
    
    len_eval = len(metric_map)
    bias = 0
    eval_index = [bias]

    for ii in range(len(dataset_delay)):
        eval_index += [
            len_eval * ii + 1 + bias, 
            len_eval * ii + 2 + bias, 
            len_eval * ii + 3 + bias,
            len_eval * ii + dataset_delay[ii][1] + bias,
            len_eval * ii + 1 + bias + 8, 
            len_eval * ii + 2 + bias + 8, 
            len_eval * ii + 3 + bias + 8,
            len_eval * ii + dataset_delay[ii][1] + bias + 8,
        ]
        
    df = old_df[eval_index]
     
    # assert 0 == 1
    
    num_cols = df.shape[1]
    num_rows = df.shape[0]
    for i in range(1, num_cols):
        df.iloc[1, i] = metric_map[df.iloc[1, i]]
    
    methods = df.iloc[2:, 0].values.tolist()
    
    json_res_data = {}
    for i in range(1, num_cols):
        if df.iloc[0, i] not in json_res_data:
            json_res_data[df.iloc[0, i]] = {}
        for j in range(2, num_rows):
            if df.iloc[j, 0] not in json_res_data[df.iloc[0, i]]:
                json_res_data[df.iloc[0, i]][df.iloc[j, 0]] = {}
            if df.iloc[j, i] == "Not Founded":
                json_res_data[df.iloc[0, i]][df.iloc[j, 0]][df.iloc[1, i]] = "N/A"
            else:
                json_res_data[df.iloc[0, i]][df.iloc[j, 0]][df.iloc[1, i]] = float(df.iloc[j, i])
    all_res_data[mod] = json_res_data
    
    
    json_res = {}
    
    # {
    # "one_by_one": {
    #     "point-wise F1": {
    #         "AR": {
    #             "AIOPS": 6,
    #             "NAB": 1,
    #             "TODS": 6,
    #             "WSD": 8,
    #             "Yahoo": 1,
    #             "UCR": 2
    #         }
    #     }
    # }
    # }
    
    for i in range(1, num_cols):
        if df.iloc[1, i] not in json_res:
            json_res[df.iloc[1, i]] = {}
            for m in methods:
                json_res[df.iloc[1, i]][m] = {}
            
        datas = np.array(df.iloc[2:, i].values)
        na_cnt = 0
        for k in range(len(datas)):
            if datas[k][0] == "N":
                datas[k] = "\0"
                na_cnt += 1
                
        
        sort_idx = np.argsort(datas)[::-1]
        new_data = np.zeros(datas.shape)
        cnt = 1
        for j in sort_idx:
            if cnt > len(methods) - na_cnt:
                new_data[j] = -1
            else:
                new_data[j] = cnt
                cnt += 1
        
        for j in range(len(methods)):
            if new_data[j] == -1:
                json_res[df.iloc[1, i]][methods[j]][df.iloc[0, i]] = "N/A"
            else:
                json_res[df.iloc[1, i]][methods[j]][df.iloc[0, i]] = int(new_data[j])
         
        # df.iloc[2:, i] = new_data.astype(int)

    # new_file_path = "./CSVs/{}_leader.csv".format(mod)
    # df.to_csv(new_file_path, header=False, index=False)
    
    # methods_num = len(methods)
    # for k1 in json_res.keys():
    #     for k3, v3 in json_res[k1].items():
    #         gold, silver, bronze, sour = 0, 0, 0, 0
            
    #         for order in v3.values():
    #             if order == 1:
    #                 gold += 1
    #             elif order == 2:
    #                 silver += 1
    #             elif order == 3:
    #                 bronze += 1
    #             elif order > methods_num - 3:
    #                 sour += 1
            
    #         total = gold + silver + bronze - sour
    #         json_res[k1][k3] = {
    #             "gold":gold, 
    #             "silver":silver, 
    #             "bronze":bronze, 
    #             "sour":sour, 
    #             "total":total
    #         }
     
    all_res[mod] = json_res
    
import json
with open("site/leader_json.json", "w") as f:
    json.dump(all_res, f, indent=4)

with open("site/detail_json.json", "w") as f:
    json.dump(all_res_data, f, indent=4)
    