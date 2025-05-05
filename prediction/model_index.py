import xgboost as xgb
import joblib
from PLEs.prediction.get_metric import*

def test_model(X_test, y_test, output_file_path, base_model, cat_index,feature_columns):

    dirpath = output_file_path + '/model.json'

    if base_model == 'XGBoost':
        X_test.columns = feature_columns
        model = xgb.Booster()
        model.load_model(dirpath)
        dtest = xgb.DMatrix(data = X_test, label=y_test, enable_categorical=True)
        pr = model.predict(dtest)
        #y_pred = np.argmax(pr, axis=1)

        if y_test.nunique()>2:
            y_pred = np.argmax(pr, axis=1)
        else:
            y_pred = (pr > 0.5).astype(int)

    elif base_model == 'RF':
        dirpath = output_file_path + '/model.pkl'
        model = joblib.load(dirpath)
        y_pred = model.predict(X_test)
        pr = model.predict_proba(X_test)

    return y_test, y_pred, pr, model
