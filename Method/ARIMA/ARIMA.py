from DataFactory import TSData
from Method.MethodInterface import BaseMethod
import pmdarima as pm
from pmdarima.arima import ndiffs

try:
    import cupy as np
    print("=== Using CUDA ===")
except ImportError:
    import numpy as np

class ARIMA(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
    
    def forecast_one_step(self):
        fc, conf_int = self.model.predict(n_periods=1, return_conf_int=True)
        return (
            fc.tolist()[0],
            np.asarray(conf_int).tolist()[0])
        
    def train_valid_phase(self, tsTrain: TSData):
        kpss_diffs = ndiffs(tsTrain.train, alpha=0.05, test='kpss', max_d=3)
        adf_diffs = ndiffs(tsTrain.train, alpha=0.05, test='adf', max_d=3)
        n_diffs = max(adf_diffs, kpss_diffs)
        
        model = pm.ARIMA((16, 1, 10))
        model.fit(tsTrain.train)
        # auto = pm.auto_arima(tsTrain.train, d=n_diffs, seasonal=False, stepwise=False,
        #              suppress_warnings=True, error_action="ignore", start_p=4, max_p=16, start_q=4, max_q=16, random=True, n_fits=50,
        #              max_order=None, trace=True)

        # print(auto.order)
        self.model = model
        
    def test_phase(self, tsData: TSData):
        forecasts = []
        confidence_intervals = []
        for new_ob in tsData.test:
            fc, conf = self.forecast_one_step()
            forecasts.append(fc)
            confidence_intervals.append(conf)

            # Updates the existing model with a small number of MLE steps
            self.model.update(new_ob)
            
        forecasts = np.array(forecasts)
        assert forecasts.shape == tsData.test.shape
        
        self.__anomaly_score = (forecasts - tsData.test) ** 2
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score