from xgboost import Booster
from xgboost import XGBRegressor
import numpy as np

class Predictor:
    def __init__(self):

        if 'self.loaded_model' in globals():
            print('')
        else:
            self.loaded_model = XGBRegressor()
            self.booster = Booster()
            self.booster.load_model('xg.model')
            self.loaded_model._Booster = self.booster


    def predict(self,ls):
        #ls = [val1,val2,val3,val4,val5,val6,val7,val8]
        #ls = ls.append(val1,val2,val3,val4,val5,val6,val7,val8)
        dt = np.array(ls)
        #If the list conains only one item add the axis dimension so the model doesn't crash
        if len(ls)==1:
            dt=dt.reshape(1,-1)
        yp=self.loaded_model.predict(dt)
        return yp.tolist()
