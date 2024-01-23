from teradataml import DataFrame, create_context
from teradatasqlalchemy.types import INTEGER, VARCHAR, CLOB
from sklearn.ensemble import RandomForestRegressor
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
from collections import OrderedDict

import sys,os
import numpy as np
import json
import base64
import dill
import base64
import pickle

import joblib


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    print("Starting training...")

    from teradataml.dbutils.filemgr import install_file,remove_file
    from teradataml.analytics.utils import display_analytic_functions

        # Install STO Python script
    try:
        remove_file (file_identifier='VIVO_AltoValorSTO', force_remove=True)
    except:
        pass
    install_file(file_identifier='VIVO_AltoValorSTO', file_path=f"./VIVO_AltoValorSTO.py", 
                is_binary=False)

    # Install pickled model
    try:
        remove_file (file_identifier='model_gbc_alt_valor', force_remove=True)
    except:
        pass
    install_file(file_identifier='model_gbc_alt_valor', file_path=f"./model_gbc_alt_valor.pickle", 
                is_binary=True)

    print("Finished training")

    # export model artefacts
    #joblib.dump(model, f"{context.artifact_output_path}/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    #xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
    #                pmml_f_name=f"{context.artifact_output_path}/model.pmml")

    print("Saved trained model")

    #from xgboost import plot_importance
    #model["xgb"].get_booster().feature_names = feature_names
    #plot_importance(model["xgb"].get_booster(), max_num_features=10)
    #save_plot("feature_importance.png", context=context)

    feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")

    print("Recording training stats")

    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=[target_name],
                          feature_importance=feature_importance,
                          context=context)
    
    print("All done!")