from typing import Type
from Analysis.Evaluations import EvalInterface, MetricInterface
from Analysis.Evaluations.MetricBase import Auroc
from Analysis.Evaluations.utils import rec_scores
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class AurocPA(EvalInterface):
    def __init__(self, figname=None) -> None:
        super().__init__()
        self.figname = figname
        
    def calc(self, scores, labels, all_label_normal) -> type[MetricInterface]:
        ## All labels are normal
        if all_label_normal:
            return Auroc(value=1)
        
        scores = rec_scores(scores=scores, labels=labels)
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
            
        return Auroc(value=auroc)