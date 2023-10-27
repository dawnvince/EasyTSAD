from dataclasses import dataclass
from . import MetricInterface

@dataclass
class F1class(MetricInterface):
    '''
    The F1class is a concrete implementation of the MetricInterface abstract class. It represents an F1 metric and provides methods for adding metric values, calculating averages, and converting the metric into a dictionary.

    Attributes:
    - name: str: The name of the F1 metric.
    - p: float = 0: The precision value of the metric (default 0).
    - r: float = 0: The recall value of the metric (default 0).
    - f1: float = 0: The F1 score value of the metric (default 0).
    - thres: float = -1: The threshold value of the metric (default -1).
    - num: int = 1: The number of metric instances (default 1).
    
    Methods:
    - add(self, other): Adds the values of another F1class instance to the current metric by summing their respective attributes.
    - avg(self): Calculates the average values of the precision, recall, and F1 score by dividing them by the num attribute.
    - to_dict(self): Converts the metric object into a dictionary representation. If num is 1, it includes the threshold value in the dictionary, otherwise, it excludes it.'''
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
    '''
    The Auroc class is a concrete implementation of the MetricInterface abstract class. It represents the Area Under the Receiver Operating Characteristic (AUROC) metric and provides methods for adding metric values, calculating averages, and converting the metric into a dictionary.

    Attributes:
    - value: float: The AUROC value.
    - name: str = "auroc": The name of the AUROC metric (default: "auroc").
    - num: int = 1: The number of metric instances (default: 1).
    
    Methods:
    - add(self, other): Adds the value of another Auroc instance to the current metric by summing their respective value attributes.
    - avg(self): Calculates the average value of the AUROC by dividing it by the num attribute.
    - to_dict(self): Converts the metric object into a dictionary representation, where the AUROC value is associated with its name.
    '''
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