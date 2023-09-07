import numpy as np


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