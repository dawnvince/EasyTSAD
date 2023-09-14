import os
import csv
import json

'''
Dataset1, Dataset2, ...
Method, PA, PA-10, PA-20, PA-30, AUROC, AUPRC

'''

base_path = "Results/EvalResult"
def get_info(method, dataset, mode):
    info_path = os.path.join(base_path, method, dataset, mode) + "_avg.json"
    if not os.path.exists(info_path):
        return [method] + ["NaN"]*6
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    info_to_csv = [method]
    info_to_csv.append(info["best f1 under pa"]["f1"])
    info_to_csv.append(info["best f1 under 10-delay pa"]["f1"])
    info_to_csv.append(info["best f1 under 20-delay pa"]["f1"])
    info_to_csv.append(info["best f1 under 30-delay pa"]["f1"])
    info_to_csv.append(info["auroc"])
    info_to_csv.append(info["auprc"])
    
    for i in range(len(info_to_csv)):
        if isinstance(info_to_csv[i], float):
            info_to_csv[i] = round(info_to_csv[i], 4)
    return info_to_csv

def get_UCR(method, mode):
    info_path = os.path.join(base_path, method, "UCR", mode) + "_avg.json"
    if not os.path.exists(info_path):
        return [method] + ["NaN"]
    with open(info_path, 'r') as f:
        info = json.load(f)
    return [method] + [round(info["Event Detected"]["detected"], 4)]
    

dataset_l = ["AIOPS", "NAB", "TODS", "WSD", "Yahoo"]
method_l = ["AE", "AnomalyTransformer", "AR", "Donut", "EncDecAD", "FCVAE", "LSTMADalpha", "LSTMADbeta", "SRCNN", "TFAD", "TimesNet"]
tt = 7

def write_to_csv(mode):
    file_name = "Results/CSV/{}.csv".format(mode)
    
    with open(file_name, 'w') as f:
        w = csv.writer(f)
        duplicated_list = []
        for item in dataset_l:
            for _ in range(tt):
                duplicated_list.append(item)
        
        duplicated_list.extend(['UCR', 'UCR'])
        w.writerow(duplicated_list)
        
        info_line = ["Method", "PA", "PA10", "PA20", "PA30", "ROC", "PRC"] * len(dataset_l) + ["Method", "Detectd"]
        w.writerow(info_line)
        
        for m in method_l:
            res = []
            for d in dataset_l:
                res += get_info(m, d, mode)
            res += get_UCR(m, mode)
            
            w.writerow(res)
            
if __name__ == "__main__":
    write_to_csv("one_by_one")
    write_to_csv("all_in_one")
    write_to_csv("transfer_within_dataset")
            
        