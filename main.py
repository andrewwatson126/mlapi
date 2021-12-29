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


app = FastAPI()

logger = logging.getLogger(__name__)


origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ROOT_FOLDER = "data/"
INPUT_FOLDER = ROOT_FOLDER + "input/"
ALGORITHM_LIST = INPUT_FOLDER + "algorithms.json"
PROJECT_FOLDER = ROOT_FOLDER + "project/"
PROJECT_LIST = PROJECT_FOLDER + "project_list.json"


# Global Variables
app.project_list = []
app.algorithms = []


models = [ 
        ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')), 
        ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
        ('KNeighborsClassifier', KNeighborsClassifier()),
        ('DecisionTreeClassifier', DecisionTreeClassifier()),
        ('GaussianNB', GaussianNB()),
        ('SVC', SVC(gamma='auto'))
    ]


# BaseModel definitions
class Project(BaseModel):
    id: int = 0
    name: str = ""  
    created_date: Optional[datetime] 
    description: Optional[str] = ""
    data_file: Optional[str] = ""
    created_by: str = "" 
    model: Optional[str] = "Supervised"
    algorithms: Optional[List[str]] = []
    features: Optional[List[str]] = []
    label: Optional[List[str]] = []
    accuracy: Optional[typing.Dict[str,List[float]]] = {}


#
# on fastapi startup
#

@app.on_event("startup")
async def startup_event():
    logger.info('Starting up fastapi server...')
    read_project_list()
    read_algorithms()
    logger.info('Started up fastapi server')

#
# /project
#

# upload file
@app.post("/projects/uploadfile")
async def upload_file(project_id: int, file: UploadFile = File(...)):
    logger.info("upload_file file=" + file.filename)
    
    path = PROJECT_FOLDER + str(project_id) 
    try:
        os.mkdir(path)
    except Exception as err:
        logging.warn("failed to cereate folder, already exists folder=",path, err)
        
    storeFile = PROJECT_FOLDER + str(project_id) + '/' + file.filename
   
    logger.info("upload_file storeFile=" + storeFile)
    
    # delete file if it exists
    #if not os.path.exists(storeFile):
    #    os.remove(storeFile)

    
    #try:
    async with aiofiles.open(storeFile, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    
    load_file(project_id, storeFile)
    project = get_project_by_id(project_id)
    project["data_file"] = file.filename
    modify_project(project)

    #except Exception as e:
    #    logger.error("upload_file()", str(e))
    #    return JSONResponse(
    #        status_code = status.HTTP_400_BAD_REQUEST,
    #        content = { 'message' : str(e) }
    #        )
    #else:
    #    return JSONResponse(
    #        status_code = status.HTTP_200_OK,
    #        content = {"result":'result'}
    #        )    


# get project list
@app.get("/projects", response_model=List[Project])
def get_project_list():
    logger.info("get_project_list", app.project_list)
    return app.project_list


# get project 
@app.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: int):
    logger.info("get_project project_id=", project_id)
    return get_project_by_id(project_id)


# create project 
@app.post("/projects")
async def create_project(project: Project):
    logger.info("create_project project=", project)

    max = get_max_project_id()
    project.id = max +1

    project_json = jsonable_encoder(project)
    app.project_list.append(project_json)
    logger.info("create_project appended=", app.project_list)
        
    store_project_list()

    return project

# update project 
@app.put("/projects")
async def update_project(project: Project):
    logger.info("update_project project=", str(project))
    p = get_project_by_id(project.id)
    # p["name"] = project.name
    # p["created_date"] = project.created_date
    # p["description"] = project.description
    # p["data_file"] = project.data_file
    # p["created_by"] = project.created_by
    p["model"] = project.model
    p["algorithms"] = project.algorithms
    p["features"] = project.features
    p["label"] = project.label
    # p["accuracy"] = project.accuracy

    store_project_list()

    return project

# delete
@app.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    logger.info("delete_project projectId=", project_id)
    for project in app.project_list:
        if project.get("id") == project_id:
            app.project_list.remove(project)
            
    store_project_list()


@app.post("/projects/predict")
def predict(project_id: int, data: list ):
    logger.info("predict(project_id=" + str(project_id) + ")")
    predictDict = {}
    project = get_project_by_id(project_id)

    for algorithm in app.algorithms:
        algorithm_label = algorithm["label"]
        logger.info("label=")
        logger.info(algorithm_label)
        model = load_model(project_id, algorithm_label)
        d = data 
        #d = [[6.1,2.9,4.7,1.4]]
        logger.info(model.predict(d))
        predictDict[algorithm_label] = str(model.predict(d))
        #predictDict.update({name: model_dict[name].predict(d)})

    return {"predictDict": predictDict }


# get correlation
@app.get("/projects/correlation/{project_id}", response_class=FileResponse)
def correlation(project_id: int):
    logger.info("correlation project_id=", str(project_id))
    project = get_project_by_id(project_id)
    
    local_data_file = PROJECT_FOLDER + str(project_id) + '/' + project["data_file"]
    df = pd.read_csv(local_data_file)

    # run correlation matrix and plot
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns_plot = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

    fig = sns_plot.get_figure()
    local_correlation_file = PROJECT_FOLDER + str(project_id) + '/' + "correlation.png"
    fig.savefig(local_correlation_file)    

    encoded_string = ""
    image_file = open(local_correlation_file, "rb")
    encoded_bytes = base64.b64encode(image_file.read() )
    #encoded_string = encoded_bytes.decode("ascii")  
    encoded_string = encoded_bytes.decode("utf-8")  
    logger.info("***************************************" + str(encoded_string))
    print("************************************" + str(encoded_string))
    image_file.close()

    local_correlation_base64_file = PROJECT_FOLDER + str(project_id) + '/' + "correlation.txt"
    base64_file = open(local_correlation_base64_file, "w")
    base64_file.write(str(encoded_string))
    base64_file.close()
    

    return local_correlation_base64_file


# get features and labels
@app.get("/projects/features_labels/{project_id}")
def get_features_and_labels(project_id: int):
    logger.info("get_features_and_labels project_id=", str(project_id))
    project = get_project_by_id(project_id)
    
    fl = []
    for feature in project["features"]:
        fl.append({ "label": feature.replace(' ','') , "name": feature.replace(' ','')})
    for label in project["label"]:
        fl.append({ "label": label.replace(' ','') , "name": label.replace(' ','')})
    
    return fl
            
#
# Algortihms
#
            
# get algorithms
@app.get("/algorithms")
def get_algorithms():
    logger.info("get_algorithms")
    return app.algorithms
 
#
# Private Methods
#    

#get max project id
def get_max_project_id():
    logger.debug("get_max_project_id()")
    max = 0
    for p in app.project_list:
        if p["id"] > max:
            max = p["id"]
    logger.debug("get_max_project_id() max=", max)
    return max


# get project by project id
def get_project_by_id(project_id: int):
    logger.info("get_project_by_id")
    for project in app.project_list:
        if project.get("id") == project_id:
            return project
        
    raise NotFoundException("Project with id , project_id,  not found")
 

def update_project_list(project):
    logger.info("update_project_list")
    for p in app.project_list:
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
    store_project_list() 
        

# Machine Learning Methods

def load_file(project_id: int, data_file_name: str):
    logger.info("load_file project_id=" + str(project_id) + " data_file_name=" + data_file_name)
    
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
    store_project_list()

    #train on a single model
    # Split-out validation dataset
    array = dataset.values
    X = array[0:,0:len(features)]
    y = array[0:,len(features)]
    print(">>>>>>>>>>>>>>>>>>>>>>> x " + str(X))
    print(">>>>>>>>>>>>>>>>>>>>>>> y " + str(y))
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

    # check algorithms
    # evaluate each model in turn
    results = []
    xnames = []
    accuracy = []
    accuracyDict = {}
    model_dict = {}
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        xnames.append(name)
        accuracy = [cv_results.mean(), cv_results.std()]
        accuracyDict[name] =  accuracy

        # create model
        m = model.fit(X_train,Y_train)
        
        store_model(project_id, name, m)
        
        model_dict[name] = m
        model_dict[name].predict([[1,7,1,1,0,0,0,1,0,11.2,2,0.02,273,5,10100,3520,561000,13.2]])
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))    

    # experiment["model_dict"] = model_dict
    project["accuracy"] = {}
    project["accuracy"] = accuracyDict
    update_project_list(project)
    store_project_list()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', {"features": features},{"label": label}, {"accuracy": accuracyDict})
    return {"features": features},{"label": label}, {"accuracy": accuracyDict}




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
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC(gamma='auto')))
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

#
# Utils
#
def modify_project(project: Project):
    logger.info("modify_project projectId=", project)
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

    store_project_list()

    return project



#
# Read and Store files
#

# Store and Read Project List 
def store_project_list():
    logger.info("store_project_list=",app.project_list)
    with open(PROJECT_LIST, 'w') as f:
        json.dump(app.project_list, f, ensure_ascii=False, indent=4)
    f.close()
    read_project_list()


def read_project_list():
    logger.info("Starting read_project_list...")
    f = open(PROJECT_LIST)
    app.project_list = json.load(f)
    logger.info("read_project_list=", app.project_list)
    f.close()

# Store Model
def store_model(project_id: int, algorithm_name: str, model):
    logger.info("store_model=project_id", project_id, " algorithm_name=", algorithm_name)

    filename = PROJECT_FOLDER + str(project_id) + '/' + algorithm_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))

# Load Model
def load_model(project_id: int, algorithm_name: str):
    logger.info("load_model=project_id", project_id, " algorithm_name=", algorithm_name)

    filename = PROJECT_FOLDER + str(project_id) + '/' + algorithm_name + '.sav'
    return pickle.load(open(filename, 'rb'))


def read_algorithms():
    logger.info("Starting read_algorithms...")
    f = open(ALGORITHM_LIST)
    app.algorithms = json.load(f)
    logger.info("read_algorithms=", app.algorithms)
    f.close()

#
# Exceptions
#

# Exceptions % Exception Handlers
class NotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        
        
@app.exception_handler(NotFoundException)
async def NotFoundException_exception_handler(request: Request, exc: NotFoundException):
    return JSONResponse(
        status_code=401,
        content={"message": f"Oops! {exc.message} did something. There goes a rainbow..."},
    )        