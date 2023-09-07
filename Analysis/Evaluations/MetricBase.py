from dataclasses import dataclass
from abc import ABCMeta,abstractmethod
from typing import Dict

class MetricInterface(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def add(self, other_metric):
        raise NotImplementedError()
    
    @abstractmethod
    def avg(self):
        raise NotImplementedError()
    
    @abstractmethod
    def to_dict(self):
        raise NotImplementedError()
    

@dataclass
class F1class(MetricInterface):
    name: str
    p: float = 0
    r: float = 0
    f1: float = 0
    thres: float = -1
    num: int = 1
    
    def add(self, other):
        self.p += other.p
        self.r += other.r
        self.f1 += other.f1
        
        self.num += 1
    
    def avg(self):
        if self.num != 0:
            self.p /= self.num
            self.r /= self.num
            self.f1 /= self.num
    
    def to_dict(self):
        if self.num == 1:
            return {
                self.name: {
                    'f1':self.f1,
                    'precision': self.p,
                    'recall': self.r,
                    'threshold' : self.thres
                }
            }
        # return the metrics after average
        else:
            return {
                self.name: {
                    'f1':self.f1,
                    'precision': self.p,
                    'recall': self.r,
                }
            }
    
@dataclass
class Auroc(MetricInterface):
    value: float
    name : str="auroc"
    num: int=1
    
    def add(self, other):
        self.value += other.value
        self.num += 1
        
    def avg(self):
        if self.num != 0:
            self.value /= self.num
        
    def to_dict(self):
        return {
            self.name: self.value
        }
        

@dataclass
class Auprc(MetricInterface):
    value: float
    name : str="auprc"
    num: int=1
    
    def add(self, other):
        self.value += other.value
        self.num += 1
        
    def avg(self):
        if self.num != 0:
            self.value /= self.num
        
    def to_dict(self):
        return {
            self.name: self.value
        }