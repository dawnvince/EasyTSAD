from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import Auprc
from ..utils import rec_scores, rec_scores_event, rec_scores_kth_event
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class EventKthPrcPA(EvalInterface):
    def __init__(self, k, mode="log", base=3) -> None:
        '''
        Using Event-based point-adjustment Auprc to evaluate the models under k-delay strategy.
        
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
         An Auprc instance (Evaluations.Metrics.Auprc), including:\n
            auprc: auprc value.
        '''
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