from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import Auroc
from ..utils import rec_scores, rec_scores_event, rec_scores_kth_event
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class EventKthRocPA(EvalInterface):
    def __init__(self, k, mode="log", base=3) -> None:
        '''
        Using Event-based point-adjustment Auroc to evaluate the models under k-delay strategy.
        
        Parameters:
            k (int): Defines the delay limit.
            mode (str): Defines the scale at which the anomaly segment is processed. \n
                One of:\n
                    - 'squeeze': View an anomaly event lasting t timestamps as one timepoint.
                    - 'log': View an anomaly event lasting t timestamps as log(t) timepoint.
                    - 'sqrt': View an anomaly event lasting t timestamps as sqrt(t) timepoint.
                    - 'raw': View an anomaly event lasting t timestamps as t timepoint.
                If using 'log', you can specify the param "base" to return the logarithm of x to the given base, 
                calculated as log(x) / log(base).
            base (int): Default is 3.
        
        '''
        super().__init__()
        self.figname = None
        self.name = "%d-th auprc under event-based pa with mode %s"%(k, mode)
        self.mode = mode
        self.base = base
        self.k = k
        
    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        '''
        Returns:
         An Auroc instance (Evaluations.Metrics.Auroc), including:\n
            auroc: auroc value.
        '''
        k = self.k + margins[0]
        
        scores, labels = rec_scores_kth_event(scores=scores, labels=labels, k=k, mode=self.mode, base=self.base)
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=labels, y_score=scores, 
                                                drop_intermediate=False)
        auroc = sklearn.metrics.auc(fpr, tpr)
        
        if math.isnan(auroc):
            auroc = 0
        
        ## plot
        if self.figname:
            prec, recall, _ = sklearn.metrics.precision_recall_curve(y_true=labels,
                                                                     probas_pred=scores)
            display = sklearn.metrics.PrecisionRecallDisplay(precision=prec, 
                                                             recall=recall)
            display.plot()
            plt.savefig(str(self.figname) + "_event_auprc.pdf")
            
        return Auroc(value=auroc, name=self.name)