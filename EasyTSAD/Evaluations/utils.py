import math
import numpy as np

All_normal_threshold = -1e10

def rec_scores(scores, labels):
    '''
    Reconstruct scores using point adjustment.
    
    Returns:
        scores - the reconstructed scores
    '''
    rec_score = scores
    ano_flag = 0
    start, end = 0, 0
    ll = len(scores)
    for i in range(ll):
        # encounter an anomaly
        if labels[i] > 0.5 and ano_flag == 0:
            ano_flag = 1
            start = i
        
        # alleviation
        elif labels[i] <= 0.5 and ano_flag == 1:
            ano_flag = 0
            end = i
            rec_score[start:end] = np.max(scores[start:end])
            
        # marked anomaly at the end of the list
        elif ano_flag == 1 and i == ll - 1:
            ano_flag = 0
            end = i + 1
            rec_score[start:end] = np.max(scores[start:end])
            
    return rec_score

def rec_scores_kth(scores, labels, k:int):
    '''
    Reconstruct scores using k-th point adjustment.
    
    Returns:
        scores - the reconstructed scores
    '''
    rec_score = scores
    ano_flag = 0
    start, end = 0, 0
    max_score = 0
    ll = len(scores)
    for i in range(ll):
        # encounter an anomaly
        if labels[i] > 0.5 and ano_flag == 0:
            ano_flag = 1
            start = i
        
        # alleviation
        elif labels[i] <= 0.5 and ano_flag == 1:
            ano_flag = 0
            end = i
            rec_score[start:end] = max_score
            max_score = 0
            
        if labels[i] > 0.5 and i - start <= k:
            max_score = max(max_score, scores[i])
        
        # marked anomaly at the end of the list
        if ano_flag == 1 and i == ll - 1:
            ano_flag = 0
            end = i + 1
            rec_score[start:end] = max_score
            max_score = 0
            
    return rec_score

def rec_scores_event(scores, labels, mode, base):
    '''
    Reconstruct scores using point adjustment.
    
    Returns:
        scores - the reconstructed scores
    '''
    ano_flag = 0
    start, end = 0, 0
    ll = len(scores)
    new_labels = []
    new_scores = []
    
    if mode == "squeeze":
        func = lambda x: 1
    elif mode == "log":
        func = lambda x: math.floor(math.log(x+base, base))
    elif mode == "sqrt":
        func = lambda x: math.floor(math.sqrt(x))
    elif mode == "raw":
        func = lambda x: x
    else:
        raise ValueError("please select correct mode.")
    
    for i in range(ll):
        # encounter an anomaly
        if labels[i] > 0.5 and ano_flag == 0:
            ano_flag = 1
            start = i
        
        # alleviation
        elif labels[i] <= 0.5 and ano_flag == 1:
            ano_flag = 0
            end = i
            cur_anomaly_len = func(end - start)
            new_scores += [np.max(scores[start:end])] * cur_anomaly_len
            new_labels += [1] * cur_anomaly_len
            
        # marked anomaly at the end of the list
        elif ano_flag == 1 and i == ll - 1:
            ano_flag = 0
            end = i + 1
            cur_anomaly_len = func(end - start)
            new_scores += [np.max(scores[start:end])] * cur_anomaly_len
            new_labels += [1] * cur_anomaly_len
        
        if labels[i] <= 0.5:
            new_labels.append(0)
            new_scores.append(scores[i])
        
    new_scores = np.array(new_scores)
    new_labels = np.array(new_labels)
    assert new_labels.ndim == 1
    assert new_labels.ndim == 1
    return new_scores, new_labels

def rec_scores_kth_event(scores, labels, k:int, mode, base):
    '''
    Reconstruct scores using k-th point adjustment.
    
    Returns:
        scores - the reconstructed scores
    '''
    ano_flag = 0
    start, end = 0, 0
    ll = len(scores)
    new_labels = []
    new_scores = []
    
    if mode == "squeeze":
        func = lambda x: 1
    elif mode == "log":
        func = lambda x: math.floor(math.log(x+base, base))
    elif mode == "sqrt":
        func = lambda x: math.floor(math.sqrt(x))
    elif mode == "raw":
        func = lambda x: x
    else:
        raise ValueError("please select correct mode.")
    
    for i in range(ll):
        # encounter an anomaly
        if labels[i] > 0.5 and ano_flag == 0:
            ano_flag = 1
            start = i
        
        # alleviation
        elif labels[i] <= 0.5 and ano_flag == 1:
            ano_flag = 0
            end = i
            cur_anomaly_len = func(end - start)
            new_scores += [np.max(scores[start:min(end, start+k)])] * cur_anomaly_len
            new_labels += [1] * cur_anomaly_len
            
        # marked anomaly at the end of the list
        if ano_flag == 1 and i == ll - 1:
            ano_flag = 0
            end = i + 1
            cur_anomaly_len = func(end - start)
            new_scores += [np.max(scores[start:min(end, start+k)])] * cur_anomaly_len
            new_labels += [1] * cur_anomaly_len
        
        if labels[i] <= 0.5:
            new_labels.append(0)
            new_scores.append(scores[i])
        
    new_scores = np.array(new_scores)
    new_labels = np.array(new_labels)
    assert new_labels.ndim == 1
    assert new_labels.ndim == 1
    return new_scores, new_labels

def f1_score(predict, actual):
    eps = 1e-15
    tp = np.sum(predict * actual)
    # tn = np.sum((1-predict) * (1-actual))
    fp = np.sum(predict * (1-actual))
    fn = np.sum((1-predict) * actual)
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(f1), float(precision), float(recall)