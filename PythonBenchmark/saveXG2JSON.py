from xgboost import Booster
from xgboost import XGBRegressor
#ev =  dft.pop('EV')

#The following code snippet loads the saved xgboost model
loaded_model = XGBRegressor()
booster = Booster()
booster.load_model('xg.model')
loaded_model._Booster = booster
loaded_model.save_model('xg.json')
