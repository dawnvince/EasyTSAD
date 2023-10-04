from typing import Type
from Analysis.Evaluations import EvalInterface, MetricInterface
from Analysis.Evaluations.MetricBase import F1class
from Analysis.Evaluations.utils import f1_score, rec_scores

import numpy as np
from utils.spot import SPOT

class PotF1(EvalInterface):
    def __init__(self, q=1e-4, level=0.98) -> None:
        super().__init__()
        self.q = q
        self.level = level
        self.name = "pot f1"
        
    def calc(self, scores, labels, all_label_normal, margins, train_score=None) -> type[MetricInterface]:
        if not isinstance(train_score, np.ndarray):
            raise TypeError("'train_score' must be type of numpy.ndarray.")
        
        sp = SPOT(self.q)
        sp.fit(train_score, scores)
        sp.initialize(level=self.level)
        
        res = sp.run(dynamic=False)
        thres = np.mean(res['thresholds'])
        
        scores = rec_scores(scores=scores, labels=labels)
        scores = scores >= thres
        labels = labels > 0.5
        
        pot_f1, precision, recall = f1_score(scores, labels)
        return F1class(
            name=self.name, 
            p=float(precision), 
            r=float(recall), 
            f1=float(pot_f1), 
            thres=float(thres)
        )