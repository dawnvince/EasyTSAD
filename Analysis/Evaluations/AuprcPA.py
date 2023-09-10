from typing import Type
from Analysis.Evaluations import EvalInterface, MetricInterface
from Analysis.Evaluations.MetricBase import Auprc
from Analysis.Evaluations.utils import rec_scores
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class AuprcPA(EvalInterface):
    def __init__(self, figname=None) -> None:
        super().__init__()
        self.figname = figname
        
    def calc(self, scores, labels, all_label_normal) -> type[MetricInterface]:
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
            
        return Auprc(value=auprc)