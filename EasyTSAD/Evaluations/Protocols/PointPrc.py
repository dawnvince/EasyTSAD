from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import Auprc
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class PointPrc(EvalInterface):
    """
    Using traditional Auprc to evaluate the models.
    """
    def __init__(self) -> None:
        super().__init__()
        self.figname = None
        self.name = "point-based auprc"
        
    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        '''
        Returns:
         An Auprc instance (Evaluations.Metrics.Auprc), including:\n
            auprc: auprc value.
        '''
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