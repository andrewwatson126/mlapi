from types import new_class
from .project import Project
from .project import NotFoundException
import shutil
import itertools

from pydantic import BaseModel
from typing import Optional
import typing
import os
import base64

from fastapi import FastAPI, Request, UploadFile, File, status
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from fastapi.encoders import jsonable_encoder

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from enum import Enum

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot

import json

import logging
import logging.config

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

import aiofiles
#from mlxtend.plotting import plot_decision_regions
from pickle import load
from pickle import dump


# Constants
ROOT_FOLDER = "data/"
INPUT_FOLDER = ROOT_FOLDER + "input/"
ALGORITHM_LIST = INPUT_FOLDER + "algorithms.json"
SCALER_FILE = "scaler.pkl"
PROJECT_FOLDER = ROOT_FOLDER + "project/"
PROJECT_LIST = PROJECT_FOLDER + "project_list.json"
ROC_FILENAME = "roc.png"

# Global Variables
project_list = []
algorithms = []

#models = [ 
#        ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')),
#        ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
#        ('KNeighborsClassifier', KNeighborsClassifier()),
#        ('DecisionTreeClassifier', DecisionTreeClassifier()),
#        ('GaussianNB', GaussianNB()),
#        ('SVC', SVC(gamma='auto'))
#    ]

models = [ 
        ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr'))
    ]

#
# Private Methods
#    

def init():
    plt.switch_backend('agg')


###############################################################################
#get max project id
###############################################################################
def get_max_project_id():
    global project_list
    print("get_max_project_id()")
    max = 0
    for p in project_list:
        if p["id"] > max:
            max = p["id"]
    print("get_max_project_id() max=" + str(max))
    return max


###############################################################################
# get project by project id
###############################################################################
def get_project_by_id(project_id: int):
    global project_list
    print("get_project_by_id project_id=" + str(project_id))
    for project in project_list:
        if project.get("id") == project_id:
            return project
        
    raise NotFoundException("Project with id=" + str(project_id) +  " not found")
 

###############################################################################
# upldate project list
###############################################################################
def update_project_list(project):
    global project_list
    print("update_project_list")
    for p in project_list:
        if  p["id"] == project["id"]:
            p["name"] = project["name"]
            p["created_date"] = project["created_date"]
            p["description"] = project["description"]
            p["data_file"] = project["data_file"]
            p["created_by"] = project["created_by"]
            p["model"] = project["model"]
            p["algorithms"] = project["algorithms"]
            p["features"] = project["features"]
            p["label"] = project["label"]
            p["accuracy"] = project["accuracy"]
    store_project_list(project_list) 
        

# Machine Learning Methods

###############################################################################
# load file
# ###############################################################################
def load_file(project_id: int, data_file_name: str):
    global project_list
    print("load_file project_id=" + str(project_id) + " data_file_name=" + data_file_name)
    
    project = get_project_by_id(project_id)
    
    # read file - filename provided from the API
    dataset = read_csv(data_file_name)


    # set supervised/unsupervised - get supervise / unsupervised from the API
    #se the supervised flag to TRUE in the experiment

    # set the features to from 0:n-2 and label to n-1
    names = list(dataset.columns)
    features = []
    features = names[0: len(names)-1]
    labels = []
    labels = names[len(names)-1: len(names)]
    label = labels[0]
    print(">>>>>>>>>>>>>>>>>>>>>>> features " + str(features))
    print(">>>>>>>>>>>>>>>>>>>>>>> len(features) " + str(len(features)))
    print(">>>>>>>>>>>>>>>>>>>>>>> label " + str(label))
    
    #set project features and labels
    project["features"] = features
    project["label"] = []
    project["label"].append(labels[0])
    print(">>>>>>>>>>>>>>>>>>>>>>>" + str(project))
    store_project_list(project_list)

    #train on a single model
    # Split-out validation dataset
    array = dataset.values
    X = array[0:,0:len(features)]
    y = array[0:,len(features)]

    accuracyDict =  model(project, X, y, True)

    project["accuracy"] = {}
    project["accuracy"] = accuracyDict
    update_project_list(project)
    store_project_list(project_list)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', {"features": features},{"label": label}, {"accuracy": accuracyDict})

    return {"features": features},{"label": label}, {"accuracy": accuracyDict}



###############################################################################
# get best model and parameters
###############################################################################
def model(project, X, y, is_store_model):
    #print("model(X=" + str(X) + "," + str(y) + ")")
    project_id = project["id"]
    std = StandardScaler()
    X = std.fit_transform(X)
    scalar_file_path = get_project_path_by_project_id(project_id) + SCALER_FILE
    dump(std, open(scalar_file_path, 'wb'))
   
    #print("poststd>>>>>>>>>>>>>>>>>>>>>>> x " + str(X))
    #print(">>>>>>>>>>>>>>>>>>>>>>> y " + str(y))
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

    # check algorithms
    # evaluate each model in turn
    results = []
    xnames = []
    accuracy = []
    accuracyDict = {}
    model_dict = {}
    for name, model in models:
        kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
        #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy', n_jobs=8)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        xnames.append(name)
        accuracy = [cv_results.mean(), cv_results.std()]
        accuracyDict[name] =  accuracy

        # create model
        m = model.fit(X_train,Y_train)
        
        if is_store_model == True:
            store_model(project_id, name, m)
        
        model_dict[name] = m
        #model_dict[name].predict([[1,7,1,1,0,0,0,1,0,11.2,2,0.02,273,5,10100,3520,561000,13.2]])
        #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))          

    return accuracyDict


###############################################################################
# get best model and parameters
###############################################################################
def best_model(project_id: int, top_n: list, start_from_index: int):
    # result =  [ { "parameters": [], "model" : "model-name", accuracy: accuracy:float } ]

    print("best_model(" + str(project_id) + "," + str(top_n) + ")")
    
    project = get_project_by_id(project_id)

    orig_data_file_path = PROJECT_FOLDER + str(project_id) + '/' + "orig_" + project["data_file"]
    dataset_orig = read_csv(orig_data_file_path)
    
    # set the features to from 0:n-2 and label to n-1
    names = list(dataset_orig.columns)

    print("name=" + str(names))
    print("len name=" + str(len(names)))

    a_list = range(0,  len(names)-1)
    print("a_list=" + str(a_list))
    all_combinations = []
    for r in top_n:
        combinations_object = itertools.combinations(a_list, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list

    best_model_path = PROJECT_FOLDER + str(project_id) + "/best_model_higher_than_095_iteration_2801_" + str(start_from_index) + ".json"

    parameters = []
    accuracies = []
    result = []
    array = dataset_orig.values
    y = array[0:,len(names)-1]
    #print("y=" + str(y))
    for i in range(len(all_combinations)):
        X = array[0:,all_combinations[i]]
        #print("*************************** all_combinations[i]=" + str(all_combinations[i]))
        #print("*************************** X=" + str(X))

        accuracy = model(project, X, y, False)
        max = get_max_accuracy(accuracy)

        if max > 0.973:
                result.append({"index": i, "parameter": all_combinations[i], "accuracies": accuracy, "max":max } ) 

        if i % 10000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("store file at " + current_time + " index=" + str(i))
            #with open(best_model_path, 'a') as f:
            #    json.dump(result, f, ensure_ascii=True, indent=4)
            #f.close()
    
    #print('result=' + str(result))
    with open(best_model_path, 'a') as f:
        json.dump(result, f, ensure_ascii=True, indent=4)
    f.close()
    return  result


def best_model1(project_id: int, top_n: int, start_from_index: int):
    # result =  [ { "parameters": [], "model" : "model-name", accuracy: accuracy:float } ]

    print("best_model(" + str(project_id) + "," + str(top_n) + ")")
    
    project = get_project_by_id(project_id)

    orig_data_file_path = PROJECT_FOLDER + str(project_id) + '/' + "orig_" + project["data_file"]
    dataset_orig = read_csv(orig_data_file_path)
    
    # set the features to from 0:n-2 and label to n-1
    names = list(dataset_orig.columns)

    print("name=" + str(names))
    print("len name=" + str(len(names)))

    a_list = range(0,  len(names)-1)
    print("a_list=" + str(a_list))
    all_combinations = []
    for r in range(2, len(names) ):
        combinations_object = itertools.combinations(a_list, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list

    #print("all_combinations=" + str(all_combinations))
    #print("len all_combinations=" + str(len(all_combinations)))
    #print("all_combinations[0]=" + str(all_combinations[0]))
    best_model_path = PROJECT_FOLDER + str(project_id) + "/best_model_higher_than_095_iteration_2801.json"

    parameters = []
    accuracies = []
    result = []
    array = dataset_orig.values
    y = array[0:,len(names)-1]
    #print("y=" + str(y))
    for i in range(len(all_combinations)):
        if (i > start_from_index):
            X = array[0:,all_combinations[i]]
            #print("*************************** all_combinations[i]=" + str(all_combinations[i]))
            #print("*************************** X=" + str(X))

            accuracy = model(project, X, y, False)
            max = get_max_accuracy(accuracy)
            #if len(result) == 0:
            #        result.append({"index": i, "parameter": all_combinations[i], "accuracies": accuracy, "max":max } ) 
            #else: 
            #    index = 0
            #    added = False
            #    for r in result:
            #        #print("max=" + str(max) + "r['max']=" + str(r["max"]))
            #       if (max >= r["max"]):
            #           #print('index=' + str(index))
            #            #print('len(restult)=' + str(len(result)))
            #            result.insert(index, {"index": i, "parameter": all_combinations[i], "accuracies": accuracy, "max":max } ) 
            #            added = True
            #            break
            #        index = index + 1
            #    if added == False:
            #        result.append({"index": i, "parameter": all_combinations[i], "accuracies": accuracy, "max":max } ) 
            #if len(result) > top_n:
            #    result.pop()

            if max > 0.973:
                   result.append({"index": i, "parameter": all_combinations[i], "accuracies": accuracy, "max":max } ) 

            if i % 100000 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("store file at " + current_time + " index=" + str(i))
                with open(best_model_path, 'a') as f:
                    json.dump(result, f, ensure_ascii=True, indent=4)
                f.close()
    
    print('result=' + str(result))
    return  result

###############################################################################
# get max accuracy 
###############################################################################
def get_max_accuracy(accuracy):

    list = []
    for a in accuracy.items():
        #print("a="+ str(a))
        list.append(a[1][0])
        #print("a[0][0]="+ str(a[1][0]))
    return max(list)

###############################################################################
# Get X single colun and y by project Id 
###############################################################################
def get_X_and_y_by_project_id(project_id):
    dataset = get_dataset_by_project_id(project_id)
    array = dataset.values
    X = array[0:,0:1]
    y = array[0:,1]
    return X,y


def get_dataset_by_project_id(project_id):
    project = get_project_by_id(project_id)    
    data_file_path = PROJECT_FOLDER + str(project_id) + '/' + project["data_file"]
    print("data_file_path=" + data_file_path)
    dataset = read_csv(data_file_path)
    return dataset



###############################################################################
# update features and label
###############################################################################
def update_features_label(project_id, features, label):
    global project_list
    print('update_features_label(' + str(project_id) + ',' +  str(features) + ',' +   str(label) + ')')
    # load data file
    project = get_project_by_id(project_id)    
    orig_data_file_path = PROJECT_FOLDER + str(project_id) + '/' + "orig_" + project["data_file"]
    print("data_file_path=" + orig_data_file_path)
    dataset_orig = read_csv(orig_data_file_path)
    
    # arrange new features and labels
    array = dataset_orig.values
    
    index = 0
    cols = []
    for column in dataset_orig.columns:
        if column in features:
            cols.append(index)
        index = index + 1
    cols.append(len(dataset_orig.columns)-1)
    print("cols=" + str(cols))
    print("features=" + str(features))
    print("label=" + str(label))

    data = array[:, cols] 
    column_names = features
    column_names.append(label[0])  
    print("column_names=" + str(column_names))
    #columns = ['PetalLengthCm','PetalWidthCm','Species']
    dataset = pd.DataFrame(data=data, columns=column_names)
    data_file_path =  PROJECT_FOLDER + str(project_id) + '/' + project["data_file"]
    dataset.to_csv(data_file_path, index=False)

    # call file
    load_file(project_id, data_file_path)
    
    return


###############################################################################
# get orig data file features and labels (projet_id)
###############################################################################
def get_orig_features_and_labels(project_id):
    project = get_project_by_id(project_id)
    orig_data_file_path = PROJECT_FOLDER + str(project_id) + '/' + "orig_" + project["data_file"]
    print("data_file_path=" + orig_data_file_path)
    dataset_orig = read_csv(orig_data_file_path)

    fl = []
    for column in dataset_orig.columns:
        fl.append({ "label": column.replace(' ','') , "name": column.replace(' ','')})
    
    return fl



###############################################################################
# load data set
###############################################################################
def load_data_set(project_id: int):
    print("load_data_set project_id=" + str(project_id))
    
    project = get_project_by_id(project_id)
    data_file_path = get_data_file_path(project)
    return pd.read_csv(data_file_path)


###############################################################################
#get data file path (project)
###############################################################################
def get_data_file_path(project):
    return  PROJECT_FOLDER + str(project['id']) + '/' + project['data_file']

###############################################################################
# get project path (project)
###############################################################################
def get_project_path(project):
    return  PROJECT_FOLDER + str(project['id']) + '/' 

###############################################################################
# get project path by project id
###############################################################################
def get_project_path_by_project_id(project_id):
    return  PROJECT_FOLDER + str(project_id) + '/' 

###############################################################################
# ml
###############################################################################
def ml():
	# Load dataset
	url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = read_csv(url, names=names)
	# Split-out validation dataset
	array = dataset.values
	X = array[:,0:4]
	y = array[:,4]
	X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
	#models.append(('LDA', LinearDiscriminantAnalysis()))
	#models.append(('KNN', KNeighborsClassifier()))
	#models.append(('CART', DecisionTreeClassifier()))
	#models.append(('NB', GaussianNB()))
	#models.append(('SVM', SVC(gamma='auto')))
	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
		cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
		results.append(cv_results)
		names.append(name)
		print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	# Compare Algorithms
	pyplot.boxplot(results, labels=names)
	pyplot.title('Algorithm Comparison')
	pyplot.show()

###############################################################################
# calculate roc
# ###############################################################################
def roc(project_id, no_of_steps):

    X,y = get_X_and_y_by_project_id(project_id)
    #steps = round(  (max(X)[0]-min(X)[0])/100)
    steps = (max(X)[0]-min(X)[0])/no_of_steps
    print("***********************************************************************steps=" + str(steps))
    thresholds = np.arange (min(X), max(X), steps)
    
    fpr = [] 
    tpr = []
    atp = []
    afp = []
    atn = []
    afn = []

    for threshold in thresholds:
        y_pred = np.where(X >= threshold, 1, 0)
        tp = fn = fp = tn = 0
        for i in range(len(y)):
            if (y[i] == 1 and y_pred[i] == 1): 
                tp = tp + 1
            if (y[i] == 1 and y_pred[i] == 0): 
                fn = fn + 1
            if (y[i] == 0 and y_pred[i] == 1): 
                fp = fp + 1
            if (y[i] == 0 and y_pred[i] == 0): 
                tn = tn + 1
               
        if ((fp + tn) == 0):
            fpr.append(0)
        else:
            fpr.append(fp / (fp + tn))

        if ((tp + fn) == 0):
            tpr.append(0)
        else:
            tpr.append(tp / (tp + fn))

        atp.append(tp)
        afp.append(fp)
        atn.append(tn)
        afn.append(fn)


    plot_roc_by_project_id(project_id, fpr, tpr)

    return tpr, fpr, atp, afp, atn, afn, thresholds    


def plot_roc_by_project_id(project_id, fpr, tpr):
    averageX = [0,1]
    averagey = [0,1]

    plt.clf()
    plt.plot(fpr,tpr, '-')
    plt.plot(averageX,averagey)
    plt.title("ROC Curve")
    plt.xlabel("FPR - Specificity FP/(FP+TN)")
    plt.ylabel("TPR - Sensitivity TP/(TP+FN)")
    plt.legend(["ROC", "Average"])
    roc_file = "roc.png"
    roc_file_path =  PROJECT_FOLDER + str(project_id) + '/' + ROC_FILENAME
    print("plot_roc_by_project_id roc_file_path=" + roc_file_path)
    plt.savefig(roc_file_path)
    return

#
# Utils
#

###############################################################################
# png to base64
###############################################################################
def png_to_base64(project_id, image_file_path, base64_file_path):
    encoded_string = ""
    image_file = open(image_file_path, "rb")
    encoded_bytes = base64.b64encode(image_file.read() )
    encoded_string = encoded_bytes.decode("utf-8")  
    image_file.close()

    base64_file = open(base64_file_path, "w")
    base64_file.write(str(encoded_string))
    base64_file.close()
    return

###############################################################################
# modify project 
###############################################################################
def modify_project(project: Project):
    global project_list
    print("modify_project projectId=", project)
    p = get_project_by_id(project["id"])
    p["name"] = project["name"]
    p["created_date"] = project["created_date"]
    p["description"] = project["description"]
    p["data_file"] = project["data_file"]
    p["created_by"] = project["created_by"]
    p["model"] = project["model"]
    p["algorithms"] = project["algorithms"]
    p["features"] = project["features"]
    p["label"] = project["label"]
    p["accuracy"] = project["accuracy"]

    store_project_list(project_list)

    return project

###############################################################################
# Store and Read Project List 
###############################################################################
def get_index_in_dataframe(df, feature):
    cols = df.columns
    return np.where(cols == feature)[0][0]


#
# Read and Store files
#

###############################################################################
# Delete Project Folder (project) 
###############################################################################
def delete_project_folder(project):
    project_folder = PROJECT_FOLDER + str(project["id"])
    shutil.rmtree(project_folder)

###############################################################################
# Store and Read Project List 
###############################################################################
def store_project_list(project_list):
    print("store_project_list=",project_list)
    with open(PROJECT_LIST, 'w') as f:
        json.dump(project_list, f, ensure_ascii=False, indent=4)
    f.close()
    return read_project_list()


###############################################################################
# read project list
###############################################################################
def read_project_list():
    global project_list
    print("Starting read_project_list...")
    f = open(PROJECT_LIST)
    project_list = json.load(f)
    #print("read_project_list=", project_list)
    f.close()
    return project_list

###############################################################################
# Store Model
###############################################################################
def store_model(project_id: int, algorithm_name: str, model):
    print("store_model=project_id", project_id, " algorithm_name=", algorithm_name)

    filename = PROJECT_FOLDER + str(project_id) + '/' + algorithm_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))

###############################################################################
# Load Model
###############################################################################
def load_model(project_id: int, algorithm_name: str):
    print("load_model=project_id", project_id, " algorithm_name=", algorithm_name)

    filename = PROJECT_FOLDER + str(project_id) + '/' + algorithm_name + '.sav'
    return pickle.load(open(filename, 'rb'))


###############################################################################
# read algorithms
###############################################################################
def read_algorithms():
    print("Starting read_algorithms...")
    f = open(ALGORITHM_LIST)
    algorithms = json.load(f)
    print("read_algorithms=", algorithms)
    f.close()
    return algorithms