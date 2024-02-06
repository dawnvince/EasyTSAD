from .model import Model
from .model import find_length
from ...DataFactory import TSData
from .. import BaseMethod
import numpy as np


class SAND(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()

    def train_valid_phase(self, tsTrain: TSData):
        pass

    def train_valid_phase_all_in_one(self, tsTrains):
        pass

    def test_phase(self, tsData: TSData):
        test_data = tsData.test.astype(float)
        window = find_length(test_data)
        if 4 * window > len(test_data):
            window = len(test_data) // 4

        model = Model(pattern_length=window, subsequence_length=4 * window)
        model.fit(test_data)
        self.__anomaly_score = model.decision_scores_

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
