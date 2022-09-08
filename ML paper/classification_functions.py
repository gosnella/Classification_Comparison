import numpy as np
import pandas as pd
import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import MinMaxScaler

import hyperopt 
from hyperopt import hp
from hyperopt.pyll import scope


def hyperoptimiser(dd:dict):

    types = [list(dd.values())[k][0] for k in range(len(dd))]

    d1 = {}
    for i,j in enumerate(types):
        tmp = list(dd.keys())[i]
        vals = list(dd.values())[i]
        if isinstance(j, str):
            d1.update({tmp : hp.choice(tmp ,vals)})
        elif isinstance(i, int):
            d1.update({tmp : scope.int(hp.quniform(tmp,vals[0],vals[1],vals[2]))})
    return(d1)



def get_dict(new_results,cat_pars):
    cv_dict = {}
    for k, v in new_results.items():
        if k in cat_pars:
            cv_dict[k] = cat_pars[k][v]
        else:
            cv_dict[k] = v
    return(cv_dict)


def classification_optimise(X, y, num_pars:dict, cat_pars:dict, 
                            classifier, max_evals=10, 
                            test_size=0.8, con_mat=False, 
                            cv=True, n_folds=5, random_state=42):
    '''
    Finds the best combination of hyperparameters using the hyperopt package.
    
    Returns the best arguments as well as the model loss.
    
    Arguments:
    
    X: full data excluding target variable.
    
    y: target variable.
    
    test_size: size of test set.
    
    random_state: random seed for number generation.
    
    parameter_space: a dictionary of all values to be assessed amongst each other.
    
    classifier: model classifier to be chose. Must ensure parameter_space values are consistent with classifier arguments.
    
    max_evals: maximum evaluations for the hyperoptimisation.
    
    con_mat: confusion matrix of best model predictions.
        
    '''
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)
    
    trials_obj = hyperopt.Trials()

    full_dict = dict(list(num_pars.items()) + list(cat_pars.items()))

    parameter_space = hyperoptimiser(full_dict)
    
    def objective(args):
        # minimises the log_loss
        rf_reg = classifier(**args, random_state=1)

        rf_reg.fit(X_train, y_train)

        y_pred = rf_reg.predict_proba(X_test)

        return(log_loss(y_test, y_pred)) 

    
    best_results = hyperopt.fmin(objective,
                                 space=parameter_space,
                                 algo=hyperopt.tpe.suggest,
                                 trials=trials_obj,
                                 max_evals=max_evals)
    new_results = {k:int(v) for k, v in best_results.items()}
    cv_dict = get_dict(new_results,cat_pars)
    
    hp_list = list(num_pars.keys()) + list(cat_pars.keys())

#     if plot==True: 
#         tpe_results= np.array([[x['result']['loss'],list(x['misc']['vals'].values())] for x in trials_obj.trials])
#         tpe_results_df=pd.DataFrame(tpe_results,
#                                    columns=hp_list)
# #         fig = tpe_results_df.plot(subplots=True,figsize=(10, 10))
#         fig = plt.figure(figsize=(10, 10))
#         tpe_results_df.plot(subplots=True)
#         plt.show()
#         plt.show()
        
    
    if cv==True:
        
        kf=StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        cv_vals = cross_val_score(classifier(**cv_dict), X, y, cv = kf)
        
        return(cv_dict, np.mean(cv_vals), trials_obj)
        
    else:    
        
        return(cv_dict,trials_obj)