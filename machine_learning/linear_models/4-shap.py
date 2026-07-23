#!/usr/bin/env python3
""" module for a function to get SHAP explainer and values """
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """ creates a SHAP explainer and computes SHAP values """
    explainer = shap.Explainer(
        model, X_train)  # returns: shap.Explainer object
    shap_values = explainer(X_test)  # returns: shap.Explanation object
    return explainer, shap_values
