from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import Auroc
from ..utils import rec_scores
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class PointRoc(EvalInterface):
    """
    Using traditional Auroc to evaluate the models.
    """
    def __init__(self) -> None:
        super().__init__()
        self.figname = None
        self.name = "point-based auroc"
        
    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        '''
        Returns:
         An Auroc instance (Evaluations.Metrics.Auroc), including:\n
            auroc: auroc value.
        '''
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