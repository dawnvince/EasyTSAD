import os
import numpy as np
from matplotlib import pyplot as plt

# method_l = ["AE", "AnomalyTransformer", "AR", "Donut", "EncDecAD", "FCVAE", "LSTMADalpha", "LSTMADbeta", "SRCNN", "TFAD", "TimesNet"]
method_l = ["AE", "AnomalyTransformer", "AR", "Donut", "FCVAE", "LSTMADalpha", "LSTMADbeta", "SRCNN", "TimesNet"]

font1 = {'size':50}
font2 = {'size':12}
label_thres = 0.5  
dpi = 2000
clip_rate = 0.998

def plot_specific_curve(curve, scores, label, save_path):
    score_len_min = 1e10
    for score in scores:
        if not score:
            continue
        if score_len_min > len(score):
            score_len_min = len(score)
    
    method_num = len(method_l)
    plt.figure(figsize=(96, 8 * (method_num + 1)))
    curve = curve[len(curve) - score_len_min:]
    curve_len = len(curve)
    x = [i for i in range(curve_len)]
    
    figs = []
    fig_curve = plt.subplot((method_num + 1), 1, 1)
    plt.plot(x, curve, label="raw curve", linewidth=0.1, color="red")
    plt.xticks([])
    plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
    figs.append(fig_curve)
    
    for i in range(method_num):
        score = scores[i]
        if not score:
            continue
        label = label[len(label) - score_len_min:]
        score = score[len(score) - score_len_min:]
    
        top_y = score[score.argsort()[int(clip_rate * len(score)) - 1]] * 3
        bottom_y = score.min() - 0.1 * (top_y - score.min())
        
        fig_m = plt.subplot((method_num + 1), 1, i+2)
        plt.plot(x, score, label=method_l[i], linewidth=0.1, color="steelblue")
        plt.ylim(bottom_y, top_y)
        plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
        
        figs.append(fig_m)
    
    
    # count anomaly segment
    ano_seg = []
    ano_flag = 0
    start, end = 0,0
    for i in x:
        if label[i] >= label_thres and ano_flag == 0:
            start = i
            ano_flag = 1
        elif label[i] < label_thres and ano_flag == 1:
            end = i
            ano_flag = 0
            ano_seg.append((start, end))
            
        if i == curve_len - 1 and label[i] > label_thres:
            end = i
            ano_seg.append((start, end))
    
    for seg in ano_seg:
        for fig in figs:
            fig.axvspan(seg[0], seg[1], alpha=1, color='pink')
    
    # plt.title("Raw Data",loc = "left", fontdict=font2)
    # plt.title("Anomaly Score",loc = "right", fontdict=font2)
    plt.savefig("{}.pdf".format(save_path), format="pdf")
    plt.close()

def aggre_scores(score_path, dataset, curve_name, mode):
    scores = []
    for method in method_l:
        score_path_t = os.path.join(score_path, method, mode, dataset, curve_name) + ".npy"
        if os.path.exists(score_path_t):
            scores.append(np.load(score_path_t))
        else:
            scores.append(None)
        
    return scores

if __name__ == "__main__":
    score_path = "Results/ScoresResult"
    dataset = "UCR"
    modes = ["all_in_one", "one_by_one"]
    # curve_names = ["0", "1", "2", '3', '4', '01', '012', '0123', '01234', '12', '123', '1234', '23', '234', '34']
    curve_names = ["1sddb40", "BIDMC1", "CIMIS44AirTemperature5"]
    
    for mode in modes:
        for curve_name in curve_names:
            label_path = os.path.join("datasets/UTS", dataset, curve_name, "test_label.npy")
            curve_path = os.path.join("datasets/UTS", dataset, curve_name, "test.npy")
            label = np.load(label_path)
            curve = np.load(curve_path)
            
            scores = aggre_scores(score_path, dataset, curve_name, mode)
            
            plot_path = "Results/PlotResult/AALL/{}_{}".format(curve_name, mode)
            plot_specific_curve(curve, scores, label, plot_path)
            
    

