import os
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import subprocess
SEED = 42
N_FOLDS = 5
# different objective for candidate models
#input data and trial numbers and save path

def objective_XGBoost(trial, dtrain,scale_pos_weight, class_unique, cat_index, DIR):

    if class_unique>2:
        param = {
            'n_jobs': 1,
            'objective': 'multi:softprob',
            'num_class':int(class_unique),
            'eval_metric': 'mlogloss',
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'enable_categorical': True,
            'verbosity': 0,
            'cat_features': cat_index,
            'learning_rate': trial.suggest_float('learning_rate',  0.0001, 0.1, log=True),
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 100.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 100.0, log=True),
            "num_boost_round": trial.suggest_int( "num_boost_round", 10, 5000),
            'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
            'max_depth': trial.suggest_int("max_depth", 3, 30),
            'gamma': trial.suggest_float("gamma", 1e-5, 1.0, log=True),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.4, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-3, 5, log=True)
        }
    else:
        param = {
            'n_jobs': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'enable_categorical': True,
            'verbosity': 0,
            'cat_features': cat_index,
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            'n_estimators': trial.suggest_int( "n_estimators", 100, 1000),
            'max_depth' : trial.suggest_int("max_depth", 3, 30),
            'gamma': trial.suggest_float("gamma", 1e-5, 1.0, log=True),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.4, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            'scale_pos_weight':scale_pos_weight,
            'random_state': 42,
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-3, 5, log=True)
        }

    print(param)
    xgb_cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        nfold=N_FOLDS,
        stratified=True,
        num_boost_round=100,
        early_stopping_rounds=20,
        seed=SEED,
        verbose_eval=False,
    )

    # Save cross-validation results.
    filepath = os.path.join(DIR, "{}.csv".format(trial.number))
    xgb_cv_results.to_csv(filepath, index=False)
    # Extract the best score.
    if class_unique>2:
        best_score = xgb_cv_results["test-mlogloss-mean"].values[-1]
    else:
        best_score = xgb_cv_results["test-auc-mean"].values[-1]

    return best_score

def  objective_RF(trial, X_train, y_train, DIR):

    max_depth = trial.suggest_int("max_depth", 2, 20)
    max_features = trial.suggest_categorical("max_features", ['sqrt', 'log2', None])
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    min_samples_split = trial.suggest_int('min_samples_split', 20, 200)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 100)

    classifier_obj = RandomForestClassifier(max_depth=max_depth,
                                            n_estimators=n_estimators,
                                            max_features=max_features,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            class_weight='balanced')

    if y_train.nunique()>2:
        score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=N_FOLDS, scoring='roc_auc_ovr')
    else:
        score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=N_FOLDS, scoring='roc_auc')

    auc = score.mean()

    return auc


