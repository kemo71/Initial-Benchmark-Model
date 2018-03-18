import os
import pandas as pd
import numpy as nd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer, , r2_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #Loading data from files
    train_dataframe = pd.read_csv("train.csv")
    test_dataframe = pd.read_csv("test.csv")

    #We concatenate train and test before one hot encoding each so that they have equivalent features so when we predict on test it does not crash
    temp = pd.get_dummies(pd.concat([train_dataframe, test_dataframe], keys=[0, 1]))
    #Here we seperate them into different dataframes
    train_data, test_data = temp.xs(0), temp.xs(1)
    #Here we remove y column from test data
    test_data = test_data.drop(['y'], axis=1)
    #Assign y as the label
    label = train_data['y']
    #Assign training data without y as features
    features = train_data.drop(['y'], axis=1)
    #prepare training and testing sets for training 
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.4, random_state=0)
    #select best parameters, these were reached after quite a while of trial and error 
    parameters = {'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [50], 'silent': [False], 'objective': ['reg:linear'], 'booster': ['dart'], 'gamma': [0], 'min_child_weight': [15], 'max_delta_step': [0], 'subsample': [1], 'colsample_bytree': [0.7], 'colsample_bylevel':[0.5], 'reg_alpha': [0.0001], 'reg_lambda': [1], 'scale_pos_weight': [1], 'base_score': [0.5], 'random_state': [2018]}
    #Needed GridSearchCV parameter for gpu usage
    params = {'tree_method': 'gpu_hist'}
    #The scoring method indicated by problem evaluation metric on the competition page
    scorer = make_scorer(r2_score)

    #Initializing the grid object with a Gradient Boosting Regressor supplied with the parameters mentioned above and with R^2 scoring method
    grid_obj = GridSearchCV(XGBRegressor(**params), param_grid=parameters, scoring=scorer)
    #Begin training
    grid_fit = grid_obj.fit(X_train, y_train)
    #We use this line to exctract the best estimator in case we used multiple hyperparameters in the grid 
    best_estimator = grid_fit.best_estimator_
    #Then we predict on the test data split that we got from the dataset features
    best_predictions = best_estimator.predict(X_test)
    #Here we register our score
    score = grid_fit.score(X_test, y_test)
    #We print the best parameters that got us our best estimator
    print("best_params_:", grid_fit.best_params_)
    #We print the score that we got
    print("score:", score)

    #test output
    test_output_predictions = best_estimator.predict(test_data)
    #Here we store the IDs from the ID column
    test_data = test_data[['ID']]
    #And here we store the predictions
    test_data['y'] = test_output_predictions
    #Here we save them to a .csv file with the same submission format indicated on the competition page
    nd.savetxt(os.path.join(os.curdir, 'submission.csv'), test_data, delimiter=',', fmt=["%.0f","%.17f"], header="ID,y", comments='')



