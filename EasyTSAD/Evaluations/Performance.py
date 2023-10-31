import numpy as np
import logging
from . import MetricInterface

logger = logging.getLogger("logger")

class Performance:
    def __init__(self, method, dataset_name, curve_name, scores, labels, margins) -> None:
        '''
        Init Performance and check if the format of scores is valid. 
        
        Notice:
         If the length of the scores is less than labels, 
         then labels are cut to labels[len(labels) - len(scores):]
         
        Params:
         - `scores` - the anomaly scores provided by methods\n
         - `labels` - the ground truth labels\n
        '''
        self.scores = scores
        self.labels = labels
        
        self.all_label_normal = np.all(self.labels < 0.5)
        if self.all_label_normal:
            logger.warning("[{}] {}: {}     All test labels are normal. SKIP this curve. <<<".format(method, dataset_name, curve_name))
            return
        
        self.margins = margins
        
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
        
        pre_margin, post_margin = margins[0], margins[1]
        if pre_margin == 0 and post_margin == 0:
            return
        
        # collect label segments
        ano_seg = []
        flag, start, l = 0, 0, len(self.labels)
        for i in range(l):
            if i == l - 1 and flag == 1:
                ano_seg.append((start, l))
            elif self.labels[i] == 1 and flag == 0:
                flag = 1
                start = i
            elif self.labels[i] == 0 and flag == 1:
                flag = 0
                ano_seg.append((start, i))
        
        ano_seg_len = len(ano_seg)
        if ano_seg_len == 0:return

        # process pre_margin
        self.labels[max(0, ano_seg[0][0] - pre_margin): ano_seg[0][0] + 1] = 1
        for i in range(1, ano_seg_len):
            self.labels[max(ano_seg[i-1][1] + 2, ano_seg[i][0] - pre_margin):ano_seg[i][0] + 1] = 1
        
        # process post_margin
        for i in range(ano_seg_len - 1):
            self.labels[ano_seg[i][1] - 1: min(ano_seg[i][1] + post_margin, ano_seg[i+1][0] - 2 - pre_margin)] = 1
        self.labels[ano_seg[-1][1] - 1: min(ano_seg[-1][1] + post_margin, l)] = 1
        
        
    def perform_eval(self, callbacks):
        if self.all_label_normal:
            return None
        res = []
        for item in callbacks:
            item_result = item.calc(self.scores.copy(), self.labels.copy(), self.margins)
            if not isinstance(item_result, MetricInterface):
                raise TypeError(
                    "Return value of func 'calc' must be inherented from MetricInterface."
                    )
            res.append(item_result)
        
        res_dict = {}
        for i in res:
            res_dict.update(i.to_dict())
            
        return res, res_dict