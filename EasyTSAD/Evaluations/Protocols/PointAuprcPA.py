from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import Auprc
from ..utils import rec_scores
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class PointAuprcPA(EvalInterface):
    def __init__(self, figname=None) -> None:
        super().__init__()
        self.figname = figname
        self.name = "auprc"
        
    def calc(self, scores, labels, all_label_normal, margins) -> type[MetricInterface]:
        ## All labels are normal
        if all_label_normal:
            return Auprc(value=1)
        
        scores = rec_scores(scores=scores, labels=labels)
        auprc = sklearn.metrics.average_precision_score(y_true=labels, 
                                                        y_score=scores, average=None)
        
        if math.isnan(auprc):
            auprc = 0
        
        ## plot
        if self.figname:
            prec, recall, _ = sklearn.metrics.precision_recall_curve(y_true=labels,
                                                                     probas_pred=scores)
            display = sklearn.metrics.PrecisionRecallDisplay(precision=prec, 
                                                             recall=recall)
            display.plot()
            plt.savefig(str(self.figname) + "_auprc.pdf")
            
        return Auprc(value=auprc, name=self.name)