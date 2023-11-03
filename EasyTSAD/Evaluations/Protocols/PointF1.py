from typing import Type
from EasyTSAD.Evaluations import MetricInterface
from .. import MetricInterface, EvalInterface
from ..Metrics import F1class
from sklearn.metrics import f1_score, precision_score, recall_score

class PointF1(EvalInterface):
    """
    Using Traditional F1 score to evaluate the models.
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = "point-wise f1"
        
    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        '''
        Returns:
         A F1class (Evaluations.Metrics.F1class), including:\n
            best_f1: the value of best f1 score;\n
            precision: corresponding precision value;\n
            recall: corresponding recall value;
        '''
        prec = precision_score(labels, scores)
        rec = recall_score(labels, scores)
        f1 = f1_score(labels, scores)
        
        return F1class(
            name=self.name,
            p=prec,
            r=rec,
            f1=f1
        )