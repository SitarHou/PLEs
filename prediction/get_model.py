from PLEs.prediction.model_search import *
import json
import joblib
from sklearn.utils.class_weight import compute_class_weight
from PLEs.prediction.get_metric import*
import optuna
#hyperparameter_optimization with optuna

def train_model(X_train, y_train, output_file_path, cat_index, base_model, n_trials, skip=True):
    """
        train model by optuna
        Parameters
        ----------
        X_train, y_train to train model
        output_file_path: save path
        base model :choice model
        n_trials: number of trials(optuna)
    """
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)
    class_unique = y_train.nunique()
    if class_unique == 2:
        N_pos = len(y_train[y_train == 1])
        N_neg = len(y_train[y_train == 0])
        scale_pos_weight = N_neg / N_pos
    else:
        class_labels = sorted(y_train.unique())
        scale_pos_weight = compute_class_weight('balanced', classes=np.array(class_labels), y=y_train)

    if base_model =='XGBoost':
        if not skip:
            if class_unique>2 :
                fixed_params = {
                    'n_jobs': 1,
                    'objective': 'multi:softprob',
                    'num_class': int(class_unique),
                    'eval_metric': 'auc',
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'enable_categorical': True,
                    'verbosity': 0,
                    'cat_features': cat_index,
                    'scale_pos_weight': scale_pos_weight.tolist()
                }
                study = optuna.create_study(direction="minimize")

            else:
                fixed_params = {
                    'n_jobs': 1,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'enable_categorical': True,
                    'verbosity': 0,
                    'cat_features': cat_index,
                    'scale_pos_weight': scale_pos_weight
                }
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            dtrain = xgb.DMatrix(data = X_train, label=y_train, enable_categorical=True)

            #study = optuna.create_study(direction="maximize")#, sampler=optuna.samplers.TPESampler(seed=42) )
            study.optimize(lambda trial:objective_XGBoost(trial, dtrain, scale_pos_weight, class_unique, cat_index, DIR = output_file_path), n_trials = n_trials)

            print("Number of finished trials: ", len(study.trials))
            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))


            best_params = study.best_params
            params = {**fixed_params, **best_params}
            with open(output_file_path+'/best_params.json', 'w') as f:
                json.dump(params, f)
        else:
            with open(output_file_path + '/best_params.json', 'r') as f:
                params = json.load(f)

        split_idx = int(len(X_train) * 0.9)

        X_train_part, X_valid = X_train[:split_idx], X_train[split_idx:]
        y_train_part, y_valid = y_train[:split_idx], y_train[split_idx:]

        dtrain = xgb.DMatrix(X_train_part, label=y_train_part)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dvalid, 'eval')], num_boost_round=500, early_stopping_rounds = 20 )
        dirpath = output_file_path + '/model.json'
        model.save_model(dirpath)
        print('fit model')

    elif base_model == 'RF':

        if not skip:
            fixed_params = {'class_weight': 'balanced'}
            if class_unique > 2:
                study = optuna.create_study(direction="minimize")
            else:
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

            study.optimize(lambda trial:objective_RF(trial, X_train, y_train,DIR = output_file_path), n_trials = n_trials)

            print("Number of finished trials: ", len(study.trials))
            print("Best trial:")
            trial = study.best_trial
            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            best_params = study.best_params
            params = {**fixed_params, **best_params}
            with open(output_file_path + '/best_params.json', 'w') as f:
                json.dump(params, f)
        else:
            with open(output_file_path + '/best_params.json', 'r') as f:
                params = json.load(f)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        dirpath = output_file_path + '/model.pkl'
        joblib.dump(model,dirpath)
        print('fit model')

    return 0