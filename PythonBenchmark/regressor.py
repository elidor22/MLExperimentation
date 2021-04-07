from xgboost import Booster
from xgboost import XGBRegressor
import numpy as np

class Predictor:
    def __init__(self):
        self.loaded_model = XGBRegressor()
        self.booster = Booster()
        self.booster.load_model('xg.model')
        self.loaded_model._Booster = self.booster

    def predict(self,val1,val2,val3,val4,val5,val6,val7,val8):
        ls = [val1,val2,val3,val4,val5,val6,val7,val8]
        #ls = ls.append(val1,val2,val3,val4,val5,val6,val7,val8)
        dt = np.array(ls)
        dt=dt.reshape(1,-1)
        yp=self.loaded_model.predict(dt)
        return yp


pred = Predictor()
print(pred.predict(2	,0	,110	,1 ,1640	,0.01	,223	,1978))
