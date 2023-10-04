from abc import ABCMeta,abstractmethod
from typing import Dict, Type
from Analysis.Evaluations import MetricInterface

class EvalInterface(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def calc(self, scores, labels, all_label_normal, margins) -> Type[MetricInterface]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_name(self):
        return self.name