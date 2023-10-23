import numpy as np
from matplotlib import pyplot as plt
import math

font1 = {'size':10}
font2 = {'size':12}
label_thres = 0.5  
dpi = 2000
clip_rate = 0.998
eps = 1e-10
inf = 1e20

linewidth = 5
    
def plot_uts_score_only(curve, score, label, save_path):
    assert len(label) >= len(score), "Score length is longer than label length."
    label = label[len(label) - len(score):]
    curve = curve[len(curve) - len(score):]
    curve_len = len(curve)
    x = [i for i in range(curve_len)]
    
    top_y = score[score.argsort()[int(clip_rate * len(score)) - 1]] * 3
    bottom_y = score.min() - 0.1 * (top_y - score.min())
    
    plt.figure(figsize=(96, 8))
    fig1 = plt.subplot(2, 1, 1)
    plt.plot(x, curve, label="raw curve", linewidth=linewidth, color="red")
    
    plt.xticks([])
    
    plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
    fig2 = plt.subplot(2, 1, 2)
    plt.plot(x, score, label="anomaly score", linewidth=linewidth, color="steelblue")
    plt.ylim(bottom_y, top_y)
    plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
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
        fig2.axvspan(seg[0], seg[1], alpha=1, color='pink')
    
    # plt.title("Raw Data",loc = "left", fontdict=font2)
    # plt.title("Anomaly Score",loc = "right", fontdict=font2)
    plt.savefig("{}.pdf".format(save_path), format="pdf")
    plt.close()
    
    
def plot_uts_score_and_yhat(curve, y_hat, score, label, save_path):
    assert len(label) >= len(score), "Score length is longer than label length."
    assert len(y_hat) >= len(score), "Score length is longer than y_hat length."
    
    y_hat = y_hat[len(y_hat) - len(score):]
    label = label[len(label) - len(score):]
    curve = curve[len(curve) - len(score):]
    curve_len = len(curve)
    x = [i for i in range(curve_len)]
    
    top_y = score[score.argsort()[int(clip_rate * len(score)) - 1]] * 3
    bottom_y = score.min() - 0.1 * (top_y - score.min())
    
    plt.figure(figsize=(96, 8))
    fig1 = plt.subplot(2, 1, 1)
    plt.plot(x, curve, label="raw curve", linewidth=linewidth, color="red")
    plt.plot(x, y_hat, label="y_hat", linewidth=linewidth, color="green")
    
    plt.xticks([])
    
    plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
    fig2 = plt.subplot(2, 1, 2)
    plt.plot(x, score, label="anomaly score", linewidth=linewidth, color="steelblue")
    plt.ylim(bottom_y, top_y)
    plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
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
        fig2.axvspan(seg[0], seg[1], alpha=1, color='pink')
    
    # plt.title("Raw Data",loc = "left", fontdict=font2)
    # plt.title("Anomaly Score",loc = "right", fontdict=font2)
    plt.savefig("{}.pdf".format(save_path), format="pdf")
    plt.close()
    
def plot_uts_summary(curve, scores, label, save_path, methods):
    score_len_min = 1e10
    for score in scores:
        if score is None:
            continue
        if score_len_min > len(score):
            score_len_min = len(score)
    
    method_num = len(methods)
    plt.figure(figsize=(24, 2 * (method_num + 1)))
    curve = curve[len(curve) - score_len_min:]
    curve_len = len(curve)
    x = [i for i in range(curve_len)]
    
    figs = []
    fig_curve = plt.subplot((method_num + 1), 1, 1)
    plt.plot(x, curve, label="raw curve", linewidth=linewidth, color="red")
    plt.xticks([])
    plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
    
    figs.append(fig_curve)
    
    for i in range(method_num):
        score = scores[i]
        if score is None:
            continue
        label = label[len(label) - score_len_min:]
        score = score[len(score) - score_len_min:]
    
        top_y = score[score.argsort()[int(clip_rate * len(score)) - 1]] * 3
        bottom_y = score.min() - 0.1 * (top_y - score.min())
        
        fig_m = plt.subplot((method_num + 1), 1, i+2)
        plt.plot(x, score, label=methods[i], linewidth=linewidth, color="steelblue")
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
    
def plot_cdf_summary(scores, labels, save_path, methods):
    method_num = len(methods)
    cols = 4
    
    ano_seg = []
    ano_flag = 0
    start, end = 0,0
    x = [i for i in range(len(labels))]
    for i in x:
        if labels[i] >= label_thres and ano_flag == 0:
            start = i
            ano_flag = 1
        elif labels[i] < label_thres and ano_flag == 1:
            end = i
            ano_flag = 0
            ano_seg.append((start, end))
            
        if i == len(labels) - 1 and labels[i] > label_thres:
            end = i + 1
            ano_seg.append((start, end))
    
    plt.figure(figsize=(24, 4 * math.ceil(method_num/cols)))
    figs = []
    for j in range(method_num):
        score = scores[j]
        if score is None:
            continue
        
        bias = len(labels) - len(score)
        label = labels[bias:]
        score_len = len(score)
        
        try:
            if len(ano_seg) == 0:
                score_ano = [inf]
            else:
                score_ano = []
                for it in ano_seg:
                    if it[1]-bias <= 0:
                        continue
                    score_ano.append(max(score[max(it[0]-bias, 0): it[1]-bias]))
                
        except Exception as e:
            print("score is ", score)
            print("ano seg is ", ano_seg)
            print("bias is ", bias)
            raise e
        
        if len(score_ano) == 0:
            score_ano = [inf]
        score_ano_avg = sum(score_ano) / len(score_ano)
        score_ano_min = min(score_ano)
            
        
        # top_score = max(score[score.argsort()[int(clip_rate * len(score)) - 1]] * 2, score_ano_avg)
        scale_coff = 0.8
        bottom_score = score.min()
        top_score = (score_ano_avg - bottom_score) * (1/scale_coff) + bottom_score
        
        score_norm = []
        for i in range(len(label)):
            if label[i] < label_thres:
                score_norm.append(score[i])
        
        step = 1000
        x = np.linspace(bottom_score, top_score, num=step)
        
        score_interv = (top_score - bottom_score + eps) / step
        y = [0] * step
        
        score_norm = np.array(score_norm)
        score_norm = (score_norm - bottom_score + eps) / score_interv 
        score_norm = np.floor(score_norm)
        for i in score_norm:
            if i < step:
                y[int(i)] += 1
            
        y = np.cumsum(np.array(y)) / len(score_norm)
        
        fig_m = plt.subplot((math.ceil(method_num/cols)), cols, j + 1)
        
        y[0] = 0
        plt.plot(x, y, label=methods[j], linewidth=linewidth, color="steelblue")
        plt.axvline(score_ano_avg, color="red")
        plt.axvline(score_ano_min, color="pink")
        plt.ylim(0.7, 1.1)
        plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
        
        figs.append(fig_m)
        
    plt.savefig("{}.pdf".format(save_path), format="pdf")
    plt.close()
    
    
def plot_distribution_each_curve(scores, thresholds, labels, save_path, methods):
    method_num = len(methods)
    cols = 4
    
    ano_seg = []
    ano_flag = 0
    start, end = 0,0
    x = [i for i in range(len(labels))]
    for i in x:
        if labels[i] >= label_thres and ano_flag == 0:
            start = i
            ano_flag = 1
        elif labels[i] < label_thres and ano_flag == 1:
            end = i
            ano_flag = 0
            ano_seg.append((start, end))
            
        if i == len(labels) - 1 and labels[i] > label_thres:
            end = i + 1
            ano_seg.append((start, end))
    
    plt.figure(figsize=(24, 4 * math.ceil(method_num/cols)))
    figs = []
    for j in range(method_num):
        score = scores[j]
        threshold = thresholds[j] + np.min(score)
        if score is None:
            continue
        
        bias = len(labels) - len(score)
        label = labels[bias:]
        
        try:
            if len(ano_seg) == 0:
                score_ano = [inf]
            else:
                score_ano = []
                for it in ano_seg:
                    if it[1]-bias <= 0:
                        continue
                    score_ano.append(np.max(score[max(it[0]-bias, 0): it[1]-bias]))
                
        except Exception as e:
            print("score is ", score)
            print("ano seg is ", ano_seg)
            print("bias is ", bias)
            raise e
        
        if len(score_ano) == 0:
            score_ano = [inf]       
            
        scale_coff = 0.666666666
        bottom_score = score.min()
        top_score = (threshold - bottom_score) * (1/scale_coff) + bottom_score
        
        score_norm = []
        for i in range(len(label)):
            if label[i] < label_thres:
                score_norm.append(score[i])
        
        step = 1000
        # x = np.linspace(bottom_score, top_score, num=step)
        
        score_interv = (top_score - bottom_score + eps) / step
        
        y = [0] * step
        score_norm = np.array(score_norm)
        score_norm = (score_norm - bottom_score) / score_interv 
        score_norm = np.floor(score_norm)
        for i in score_norm:
            if i < step:
                y[int(i)] += 1  
        y = np.cumsum(y)
        y = y / len(score_norm)
        y[0] = 0
        
        z = [0] * step
        score_ano = np.array(score_ano)
        score_ano = (score_ano - bottom_score) / score_interv 
        score_ano = np.floor(score_ano)
        score_ano = np.clip(score_ano, a_min=0, a_max=step+10)
        for i in score_ano:
            if i < step:
                z[int(i)] += 1

        z = np.cumsum(z)
        z = z / len(score_ano)
        z[0] = 0
        
        x = [i for i in range(step)]
        
        fig_y = plt.subplot((math.ceil(method_num/cols)), cols, j + 1)
        fig_y.set_title(methods[j], y=-0.15)
        
        plt.plot(x, y, label="normality", linewidth=linewidth, color="steelblue")
        plt.axvline(scale_coff * step - 1, color="pink")
        # plt.axvline(threshold, color="pink")
        # plt.ylim(0.7, 1.1)
        # plt.legend(loc="upper left",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
        
        # fig_z = plt.twinx(fig_y)
        plt.plot(x, z, label="anomaly", linewidth=linewidth, color="red")
        plt.legend(loc="lower right",prop=font1)
        # plt.legend(loc="upper right",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
        
        figs.append(fig_y)
        
    plt.savefig("{}.pdf".format(save_path), format="pdf")
    plt.close()
    
def plot_distribution_datasets(scores_dict, thresholds_dict, labels_dict, save_path, methods, score_criterion="ano_quantile"):
    quantile = 0.998
    ano_quantile = 0.2
    method_num = len(methods)
    cols = 4
    plt.figure(figsize=(24, 4 * math.ceil(method_num/cols)))
    figs = []
    
    step = 1000
    ano_step = step
    
    ano_seg_dict = {}
    
    for curve_name, label_raw in labels_dict.items():
        if label_raw is None:
            continue
        ll = len(label_raw)
        ano_flag = 0
        start, end = 0,0
        ano_seg = []
        
        for i in range(ll):
            if label_raw[i] >= label_thres and ano_flag == 0:
                start = i
                ano_flag = 1
            elif label_raw[i] < label_thres and ano_flag == 1:
                end = i
                ano_flag = 0
                ano_seg.append((start, end))
                
            if i == ll - 1 and label_raw[i] > label_thres:
                end = i + 1
                ano_seg.append((start, end)) 
                
        ano_seg_dict[curve_name] = ano_seg
    
    cnt = 0
    
    for method in methods:
        y = [0] * step
        z = [0] * ano_step
        cnt += 1
        norm_score_all = 0
        ano_score_all = 0
        for curve_name, label_raw in labels_dict.items():
            if label_raw is None:
                continue
            score = scores_dict[method][curve_name]
            threshold = thresholds_dict[method][curve_name]
            if threshold == 0:
                continue
            else:
                threshold += np.min(score)
            
            if score is None:
                continue
            bias = len(label_raw) - len(score)
            label = label_raw[bias:]
            
            try:
                if len(ano_seg_dict[curve_name]) == 0:
                    score_ano = [inf]
                    continue
                else:
                    score_ano = []
                    for it in ano_seg_dict[curve_name]:
                        if it[1]-bias <= 0:
                            continue
                        score_ano.append(np.max(score[max(it[0]-bias, 0): it[1]-bias]))
                    
            except Exception as e:
                print("score is ", score)
                print("ano seg is ", ano_seg_dict[curve_name])
                print("bias is ", bias)
                raise e 
            
            if len(score_ano) == 0:
                continue
            scale_coff = 0.66
            bottom_score = np.min(score)
            
            score_norm = []
            for i in range(len(label)):
                if label[i] < label_thres:
                    score_norm.append(score[i])
            
            if score_criterion == "quantile":
                top_score = score_norm[np.argsort(np.array(score_norm))[int(len(score_norm)*quantile)]] + eps
            elif score_criterion == "ano_quantile":
                top_score = score_ano[np.argsort(np.array(score_ano))[int(len(score_ano)*ano_quantile)]] + eps
            else:
                top_score = (threshold - bottom_score) * (1/scale_coff) + bottom_score
            try:
                assert top_score > bottom_score
            except Exception as e:
                print(top_score, bottom_score)
                print(curve_name, method)
                raise e
                    
            # x = np.linspace(bottom_score, top_score, num=step)
            #  score_interv = (top_score - bottom_score) / step
            score_norm = np.array(score_norm).flatten()
            score_norm = (score_norm - bottom_score) * step / (top_score - bottom_score) 
            score_norm = np.floor(score_norm)
            norm_score_all += len(score_norm)
            for i in score_norm:
                if i < 0:
                    assert("invalid I")
                if i < step:
                    y[int(i)] += 1  
                    
            #  ano_score_interv = (top_score - bottom_score) / ano_step
            score_ano = np.array(score_ano).flatten()
            score_ano = (score_ano - bottom_score) * ano_step / (top_score - bottom_score) 
            # score_ano = np.floor(score_ano)
            score_ano = np.clip(score_ano, a_min=0, a_max=ano_step + 10)
            score_ano = np.round(score_ano, 2)
            ano_score_all += len(score_ano)
            for i in score_ano:
                if i < 0:
                    assert("invalid I")
                if i < ano_step:
                    i = int(i)
                    z[i] = z[i] + 1
                    
        xy = [i for i in range(step)]
        y = np.array(y) 
        y = np.cumsum(y) / norm_score_all 
        y[0] = 0
        
        # xz = [i for i in range(ano_step)]
        z = np.array(z)
        z = np.cumsum(z) / ano_score_all
        z[0] = 0
        
        fig_y = plt.subplot((math.ceil(method_num/cols)), cols, cnt)
        fig_y.set_title(method, y=-0.2)
        
        plt.plot(xy, y, label="normality", linewidth=linewidth, color="steelblue")
        # plt.axvline(scale_coff * step, color="pink")
        # plt.ylim(-0.1, 1.05)
        
        # fig_z = plt.twinx(fig_y)
        plt.plot(xy, z, label="anomaly", linewidth=linewidth, color="red")
        plt.legend(loc="lower right",prop=font1)
        # plt.legend(loc="upper right",prop=font1, handlelength=1, borderpad=0.1, handletextpad=0.3, ncol=2, columnspacing=0.5, borderaxespad=0.1)
        # plt.ylim(-0.1, 1.05)
        figs.append(fig_y)
        
    plt.savefig("{}.pdf".format(save_path), format="pdf")
    plt.close()