"""
Modelos de referencia con hiperparámetros fijos (sin tuning).

Se usan en los diagnósticos (curva de sondas, curva de aprendizaje) para que las
curvas reflejen solo el efecto de la variable estudiada y no del tuning.
"""
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from datos import RANDOM_STATE


def modelos_referencia():
    return {
        "ElasticNet": Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "SVR (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(C=20.0, gamma='scale', epsilon=0.1))
        ]),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4, num_leaves=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=2, verbose=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=2
        ),
    }
