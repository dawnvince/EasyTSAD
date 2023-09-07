import numpy as np
from matplotlib import pyplot as plt

font1 = {'size':10}
font2 = {'size':12}
label_thres = 0.5  
dpi = 2000
    
def plot_uts_score_only(curve, score, label, save_path):
    assert len(label) >= len(score), "Score length is longer than label length."
    label = label[len(label) - len(score):]
    curve = curve[len(curve) - len(score):]
    curve_len = len(curve)
    x = [i for i in range(curve_len)]
    
    plt.figure(figsize=(32, 2))
    fig1 = plt.subplot()
    plt.plot(x, curve, label="raw curve", linewidth=0.1, color="red")
    plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
    fig2 = fig1.twinx()
    plt.plot(x, score, label="anomaly score", linewidth=0.1, color="steelblue")
    plt.legend(loc="upper right",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
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
        fig1.axvspan(seg[0], seg[1], alpha=1, color='pink')
    
    plt.title("Raw Data",loc = "left", fontdict=font2)
    plt.title("Anomaly Score",loc = "right", fontdict=font2)
    plt.savefig("{}.pdf".format(save_path), dpi=dpi, format="pdf")
    
    
def plot_uts_score_and_yhat(curve, y_hat, score, label, save_path):
    assert len(label) >= len(score), "Score length is longer than label length."
    assert len(y_hat) >= len(score), "Score length is longer than y_hat length."
    
    y_hat = y_hat[len(y_hat) - len(score):]
    label = label[len(label) - len(score):]
    curve = curve[len(curve) - len(score):]
    curve_len = len(curve)
    x = [i for i in range(curve_len)]
    
    plt.figure(figsize=(16, 2))
    fig1 = plt.subplot()
    plt.plot(x, curve, label="raw curve", linewidth=0.1, color="red")
    plt.plot(x, y_hat, label="y_hat", linewidth=0.1, color="green")
    plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
    fig2 = fig1.twinx()
    plt.plot(x, score, label="anomaly score", linewidth=0.1, color="steelblue")
    plt.legend(loc="upper right",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
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
        fig1.axvspan(seg[0], seg[1], alpha=1, color='pink')
    
    plt.title("Raw Data",loc = "left", fontdict=font2)
    plt.title("Anomaly Score",loc = "right", fontdict=font2)
    plt.savefig("{}.pdf".format(save_path), dpi=dpi, format="pdf")