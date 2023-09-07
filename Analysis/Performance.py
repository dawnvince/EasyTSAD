import numpy as np
from Analysis.Evaluations import MetricInterface

class Performance:
    def __init__(self, scores, labels) -> None:
        '''
        Init Performance and check if the format of scores is valid.
        
        Notice:
         If the length of the scores is less than labels, 
         then labels are cut to labels[len(labels) - len(scores):]
         
        Params:
         scores - the anomaly scores provided by methods\n
         labels - the ground truth labels\n
        '''
        self.scores = scores
        self.labels = labels
        
        self.all_label_normal = np.all(self.labels < 0.5)
        
        try:
            if not isinstance(self.scores, np.ndarray):
                raise TypeError("Invalid scores type. Make sure that scores are np.ndarray\n")
                # return False, "Invalid scores type. Make sure that scores are np.ndarray\n"
            if self.scores.ndim != 1:
                raise ValueError("Invalid scores dimension, the dimension must be 1.\n")
                # return False, "Invalid scores dimension, the dimension must be 1.\n"
            if len(self.scores) > len(self.labels):
                raise AssertionError("Score length must less than label length! Score length: {}; Label length: {}".format(len(self.scores), len(self.labels)))
            self.labels = self.labels[len(self.labels) - len(self.scores):]
            
            # avoid negative value in scores
            self.scores = self.scores - self.scores.min()
            assert len(self.scores) == len(self.labels)
            
        except Exception as e:
            # return False, traceback.format_exc()
            raise e
        
    def do_eval(self, callbacks):
        res = []
        for item in callbacks:
            item_result = item.calc(self.scores, self.labels, self.all_label_normal)
            if not isinstance(item_result, MetricInterface):
                raise TypeError(
                    "Return value of func 'calc' must be inherented from MetricInterface."
                    )
            res.append(item_result)
        
        res_dict = {}
        for i in res:
            res_dict.update(i.to_dict())
            
        return res, res_dict