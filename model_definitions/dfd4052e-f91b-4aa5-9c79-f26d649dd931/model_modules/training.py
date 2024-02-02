import teradataml
from teradataml import DataFrame
from teradataml import *
from teradataml import create_context, get_context, remove_context, execute_sql

from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib

def plot_feature_importance(fi, img_filename):
    import pandas as pd
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()
    

def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    #ULTIMA  target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    #ULTIMA  train_df = DataFrame.from_query(context.dataset_info.sql)
    #ULTIMA  train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    #ULTIMA  X_train = train_pdf[feature_names]
    #ULTIMA  y_train = train_pdf[target_name]

    print("Starting training...")

    # fit model to training data
    #model = Pipeline([('scaler', MinMaxScaler()),
    #                  ('xgb', XGBClassifier(eta=context.hyperparams["eta"],
    #                                        max_depth=context.hyperparams["max_depth"]))])

    #model.fit(X_train, y_train)

    print("Finished training") 
    print ()

    #print(context.dataset_info.predictions_database)
    
    print("Inicia Install")
    
    #install_file(file_identifier='VIVO_AltoValorSTO', file_path=f"./VIVO_AltoValorSTO.py", 
    #         is_binary=False)
    
    #execute_sql("call SYSUIF.INSTALL_FILE('stoSalesForecastnew', 'stoSalesForecastnew.py', 'cz!./stoSalesForecastnew.py');")
    
  
    # export model artefacts
    #joblib.dump(model, f"{context.artifact_output_path}/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    #xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
    #                pmml_f_name=f"{context.artifact_output_path}/model.pmml")

    print("Saved trained model")

    #from xgboost import plot_importance
    #model["xgb"].get_booster().feature_names = feature_names
    
    #plot_importance(model["xgb"].get_booster(), max_num_features=10)
    save_plot("feature_importance.png", context=context)

    #feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")

    print("Inicia Nuevo")
    
    #model_pdf = model.result.to_pandas()[['Id','Score']]
    #predictor_dict = {}
    
    #for index, row in model_pdf.iterrows():
    #    if row['predictor'] in feature_names:
    #        value = row['estimate']
    #        predictor_dict[row['predictor']] = value

    #feature_importance = dict(sorted(predictor_dict.items(), key=lambda x: x[1], reverse=True))
    #keys, values = zip(*feature_importance.items())
    #norm_values = (values-np.min(values))/(np.max(values)-np.min(values))
    #feature_importance = {keys[i]: float(norm_values[i]*1000) for i in range(len(keys))}
    #plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")


    
    #record_training_stats(train_df,
    #                      features=feature_names,
    #                      targets=[target_name],
    #                      categorical=[target_name],
    #                      feature_importance=feature_importance,
    #                      context=context)
