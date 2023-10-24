from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import Auprc
from ..utils import rec_scores, rec_scores_event, rec_scores_kth_event
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class EventKthPrcPA(EvalInterface):
    def __init__(self, k, mode="log", base=3, figname=None) -> None:
        super().__init__()
        self.figname = figname
        self.name = "%d-th auprc under event-based pa with mode %s"%(k, mode)
        self.mode = mode
        self.base = base
        self.k = k
        
    def calc(self, scores, labels, all_label_normal, margins) -> type[MetricInterface]:
        ## All labels are normal
        if all_label_normal:
            return Auprc(value=1, name=self.name)
        
        k = self.k + margins[0]
        
        scores, labels = rec_scores_kth_event(scores=scores, labels=labels, k=k, mode=self.mode, base=self.base)
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
            plt.savefig(str(self.figname) + "_event_auprc.pdf")
            
        return Auprc(value=auprc, name=self.name)