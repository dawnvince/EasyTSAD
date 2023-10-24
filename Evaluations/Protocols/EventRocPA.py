from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import Auroc
from ..utils import rec_scores, rec_scores_event, rec_scores_kth_event
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class EventRocPA(EvalInterface):
    def __init__(self, mode="log", base=3, figname=None) -> None:
        super().__init__()
        self.figname = figname
        self.name = "event-based auroc under pa with mode %s"%(mode)
        self.mode = mode
        self.base = base
        
    def calc(self, scores, labels, all_label_normal, margins) -> type[MetricInterface]:
        ## All labels are normal
        if all_label_normal:
            return Auroc(value=1, name=self.name)
        
        scores, labels = rec_scores_event(scores=scores, labels=labels, mode=self.mode, base=self.base)
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=labels, y_score=scores, 
                                                drop_intermediate=False)
        auroc = sklearn.metrics.auc(fpr, tpr)
        
        if math.isnan(auroc):
            auroc = 0
        
        ## plot
        if self.figname:
            display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc, 
                                                      estimator_name='AUROC under PA')
            display.plot()
            plt.savefig(str(self.figname) + "_auroc.pdf")
            
        return Auroc(value=auroc, name=self.name)