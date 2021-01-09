# PREDICTION SCRIPT FOR HOME CREDIT DEFAULT PREDICTION

import os
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='prediction_type', action='store_true')
parser.add_argument('--test', dest='prediction_type', action='store_false')
parser.set_defaults(prediction_type=True)
parser.add_argument("--return_counts", type=bool, default=True)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=61723)
args = parser.parse_args()

#usage: pydevconsole.py [-h] [--train] [--test]
#pydevconsole.py: error: unrecognized arguments: --mode=client --port=61790


final_train = pd.read_pickle("homecredit_final/Final/final_train_df.pkl")
final_test = pd.read_pickle("homecredit_final/Final/final_test_df.pkl")

feats = [f for f in final_test.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index',
                                                    "APP_index", "BURO_index", "PREV_index", "INSTAL_index",
                                                    "CC_index", "POS_index"]]

if args.prediction_type:
    y_train = final_train["TARGET"]
    x_train = final_train[feats]
    #c.fit(x_train.values.reshape(-1, 1), y_train)

    cur_dir = os.getcwd()
    os.chdir('homecredit_final/Final/')
    model = pickle.load(open('lightgbm_final_model.pkl', 'rb'))
    os.chdir(cur_dir)

    y_pred = model.predict_proba(x_train)[:, 1]
    print("TRAIN AUC SCORE:", roc_auc_score(y_train, y_pred))
else:
    x_test = final_test[feats]
    cur_dir = os.getcwd()
    os.chdir('homecredit_final/Final/')
    model = pickle.load(open('lightgbm_final_model.pkl', 'rb'))
    os.chdir(cur_dir)
    y_pred = model.predict_proba(x_test)[:, 1]
    ids = final_test['SK_ID_CURR']
    submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': y_pred})
    os.chdir('homecredit_final/Final')
    submission.to_csv("sub_from_prediction_py.csv", index=False)
    print("Submission file has been created in:", "models/lightgbm_final_model.pkl")

# ValueError: Number of features of the model must match the input. Model n_features_ is 765 and input n_features is 754
