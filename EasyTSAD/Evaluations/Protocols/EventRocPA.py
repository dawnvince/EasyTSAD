from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import Auroc
from ..utils import rec_scores, rec_scores_event, rec_scores_kth_event
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class EventRocPA(EvalInterface):
    def __init__(self, mode="log", base=3, figname=None) -> None:
        """
        Using Event-based point-adjustment Auroc to evaluate the models.
        
        Parameters:
            mode (str): Defines the scale at which the anomaly segment is processed. \n
                One of:\n
                    - 'squeeze': View an anomaly event lasting t timestamps as one timepoint.
                    - 'log': View an anomaly event lasting t timestamps as log(t) timepoint.
                    - 'sqrt': View an anomaly event lasting t timestamps as sqrt(t) timepoint.
                    - 'raw': View an anomaly event lasting t timestamps as t timepoint.
                If using 'log', you can specify the param "base" to return the logarithm of x to the given base, 
                calculated as log(x) / log(base).
            base (int): Default is 3.
        """
        super().__init__()
        self.figname = figname
        self.name = "event-based auroc under pa with mode %s"%(mode)
        self.mode = mode
        self.base = base
        
    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        '''
        Returns:
         An Auroc instance (Evaluations.Metrics.Auroc), including:\n
            auroc: auroc value.
        '''
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