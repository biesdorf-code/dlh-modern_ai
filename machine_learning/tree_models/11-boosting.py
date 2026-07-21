#!/usr/bin/env python3
""" module that builds boosting classifiers """
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """ Returns an untrained boosting classifier matching name """

    if name == 'adaboost':
        return ensemble.AdaBoostClassifier(n_estimators=n_estimators,
                                           random_state=random_state)
    if name == 'gradientboosting':
        return ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state)
    if name == 'xgboost':
        return xgb.XGBClassifier(n_estimators=n_estimators,
                                 random_state=random_state)
    if name == 'lightgbm':
        return lgb.LGBMClassifier(n_estimators=n_estimators,
                                  random_state=random_state,
                                  verbose=-1)

    raise ValueError(f"Unknown model name '{name}'")
