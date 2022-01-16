from . import project 
from . import util
import math


from typing import Optional
import typing
import os
import base64

from scipy.stats import spearmanr

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

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
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


app = FastAPI()

logger = logging.getLogger(__name__)


#origins = [
#    "http://127.0.0.1:3000",
#    "http://localhost:3000",
#    "http://127.0.0.1:8123",
#    "http://localhost:8123",
#]

origins = origins = ['*']


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
SCALER_FILE = "scaler.pkl"
PROJECT_FOLDER = ROOT_FOLDER + "project/"
PROJECT_LIST = PROJECT_FOLDER + "project_list.json"
ROC_FILENAME = "roc.png"


# Global Variables
app.project_list = []
app.algorithms = []


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
    print('Starting up fastapi server...')
    app.project_list = util.read_project_list()
    app.algorithms = util.read_algorithms()
    util.init()
    print('Started up fastapi server')

#
# /project
#

###############################################################################
# get project list
###############################################################################
@app.get("/projects", response_model=List[Project])
def get_project_list():
    print("get_project_list", app.project_list)
    return app.project_list


###############################################################################
# get project 
###############################################################################
@app.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: int):
    print("get_project project_id=", project_id)
    return util.get_project_by_id(project_id)


###############################################################################
# create project 
###############################################################################
@app.post("/projects")
async def create_project(project: Project):
    print("create_project project=", project)

    max = util.get_max_project_id()
    project.id = max +1

    project_json = jsonable_encoder(project)
    app.project_list.append(project_json)
    print("create_project appended=", app.project_list)
        
    app.project_list = util.store_project_list(app.project_list)

    return project

# update project 
@app.put("/projects")
async def update_project(project: Project):
    print("update_project project=", str(project))
    p = util.get_project_by_id(project.id)
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

    app.project_list = util.store_project_list(app.project_list)
    
    util.update_features_label(project.id, project.features, project.label)

    return project

###############################################################################
# delete project
###############################################################################
@app.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    print("delete_project projectId=", project_id)
    for project in app.project_list:
        if project.get("id") == project_id:
            app.project_list.remove(project)
            
    util.store_project_list(app.project_list)
    util.delete_project_folder(project)
    

###############################################################################
# upload file
###############################################################################
@app.post("/projects/uploadfile")
async def upload_file(project_id: int, file: UploadFile = File(...)):
    print("upload_file file=" + file.filename)
    
    path = PROJECT_FOLDER + str(project_id) 
    try:
        os.mkdir(path)
    except Exception as err:
        logging.warn("failed to cereate folder, already exists folder=",path, err)
        
    storeFile = PROJECT_FOLDER + str(project_id) + '/' + file.filename
    print("upload_file storeFile=" + storeFile)
    
    # delete file if it exists
    #if not os.path.exists(storeFile):
    #    os.remove(storeFile)

    
    #try:
    async with aiofiles.open(storeFile, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    
    # save also as orig file
    project = util.get_project_by_id(project_id)
    orig_data_file_path = PROJECT_FOLDER + str(project_id) + '/' + "orig_" + file.filename
    print("upload_file save orig file to " + orig_data_file_path)
    dataset_orig = read_csv(storeFile)
    dataset_orig.to_csv(orig_data_file_path, index=False)
        
    
    #try: 
    util.load_file(project_id, storeFile)
    project = util.get_project_by_id(project_id)
    project["data_file"] = file.filename
    util.modify_project(project)
    app.project_list = util.read_project_list()

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


###############################################################################
# Predict
###############################################################################
@app.post("/projects/predict")
def predict(project_id: int, data: list ):
    print("predict(project_id=" + str(project_id) + ")")
    predictDict = {}
    project = util.get_project_by_id(project_id)

    for algorithm in app.algorithms:
        algorithm_label = algorithm["label"]
        print("label=")
        print(algorithm_label)
        model = util.load_model(project_id, algorithm_label)
        d = data 
        d2 = d[0]
        d3 = [float(i) for i in d2]
        d = []
        d.append(d3)
        #d.append([1.0,1.0,1.0,1.0])
        #d.append([5.0,5.0,5.0,5.0])
        scalar_file_path = util.get_project_path_by_project_id(project_id) + SCALER_FILE
        print("scalar_file_path=" + scalar_file_path)
        std = load(open(scalar_file_path, 'rb'))
        print("predata=" + str(d))
        d = std.transform(d)
        print("postdata=" + str(d))
        d1 = []
        d1.append(d[0])
        d = d1
        print("postdata2=" + str(d))
        print("label=")
        #d = [[6.1,2.9,4.7,1.4]]
        print(model.predict(d))
        predictDict[algorithm_label] = str(model.predict(d))
        #predictDict.update({name: model_dict[name].predict(d)})

    return {"predictDict": predictDict }


###############################################################################
# get correlation plot
###############################################################################
@app.get("/projects/correlation/{project_id}", response_class=FileResponse)
def correlation(project_id: int):
    print("correlation project_id=", str(project_id))
    project = util.get_project_by_id(project_id)
    
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
    encoded_string = encoded_bytes.decode("utf-8")  
    print("***************************************" + str(encoded_string))
    print("************************************" + str(encoded_string))
    image_file.close()

    local_correlation_base64_file = PROJECT_FOLDER + str(project_id) + '/' + "correlation.txt"
    base64_file = open(local_correlation_base64_file, "w")
    base64_file.write(str(encoded_string))
    base64_file.close()
    

    return local_correlation_base64_file


###############################################################################
# get best model and parameters
###############################################################################
@app.get("/projects/best_model/{project_id}")             
def best_model(project_id: int, top_n: int, start_from_index: int):
    # result =  [ { "parameters": [], "model" : "model-name", accuracy: accuracy:float } ]
    result = util.best_model(project_id, top_n, start_from_index)
    return result

###############################################################################
# get correlation values
###############################################################################
@app.get("/projects/correlation_values/{project_id}")
def correlation(project_id: int):
    print("correlation project_id=", str(project_id))
    project = util.get_project_by_id(project_id)
    
    local_data_file = PROJECT_FOLDER + str(project_id) + '/' + project["data_file"]
    df = pd.read_csv(local_data_file)

    result = []
    
    # calculate correlations
    fs = project["features"].copy()
    print("*******************************")
    print("features=" + str(fs))
    print("*******************************")
    array = df.values
    id = 0
    for f1 in project["features"]:
        fs.remove(f1)
        print('>>>' + str(fs))
        for f2 in fs:
            index_f1 = util.get_index_in_dataframe(df, f1)
            index_f2 = util.get_index_in_dataframe(df, f2)
            array_f1 = array[0:,index_f1]
            array_f2 = array[0:,index_f2]
            print("array_f1=" + str(array_f1))
            print("array_f2=" + str(array_f2))
            #corr = matthews_corrcoef(array_f1, array_f2)
            corr, _ = spearmanr(array_f1, array_f2)

            if math.isnan(corr):
                corr = -1

            print(f1 + " - "  + f2 + " = " + str(corr))
            id = id + 1
            result.append({"feature1": f1, "feature2": f2, "correlation": corr})

    print("result=" + str(result))
    return result

###############################################################################
# get ROC
###############################################################################
@app.get("/projects/roc/{project_id}")
def roc(project_id: int, no_of_steps: int):
    print("roc project_id=", str(project_id))

    # only one feature and one label must be selected 
    # the label values must be 0s and 1s

    tpr, fpr, atp, afp, atn, afn, thresholds = util.roc(project_id, no_of_steps) 

    print("tpr=" + str(tpr))

    print("atp=" + str(atp))

    auc_value = auc(fpr, tpr)
    print("auc_value=" + str(auc_value))

    # outcome = util.get_label_count(project_id)

    result = []
    for i in range(len(tpr)):
        result.append({
            "thresholds":  thresholds[i], 
            "sensitivity": tpr[i], 
            "specificity": fpr[i],
            "tp": atp[i],
            "fp": afp[i],
            "tn": atn[i],
            "fn": afn[i]
            })

    return { "result" : result, "auc": auc_value }
    


###############################################################################
# get ROC plot file
###############################################################################
@app.get("/projects/roc_plot_file/{project_id}", response_class=FileResponse)
def get_roc_plot_file(project_id: int):
    logger.info("get_roc_plot_file project_id=", str(project_id))
    project = util.get_project_by_id(project_id)
    roc_file_path =  PROJECT_FOLDER + str(project_id) + '/' + ROC_FILENAME
    base64_roc_file_path = roc_file_path.replace("png","txt")

    util.png_to_base64(project_id, roc_file_path, base64_roc_file_path)

    return base64_roc_file_path


###############################################################################
# get features and labels
###############################################################################
@app.get("/projects/features_labels/{project_id}")
def get_features_and_labels(project_id: int):
    print("get_features_and_labels project_id=", str(project_id))
    project = util.get_project_by_id(project_id)
    
    fl = util.get_orig_features_and_labels(project_id)
    return fl
        
###############################################################################
# plot model (project_id)
###############################################################################
@app.get("/projects/plot/{project_id}")
def plot_data(project_id: int, algorithm: str):
    logger.info("plot_model project_id=", str(project_id), " algorithm=", algorithm)
    project = util.get_project_by_id(project_id)
    project_path = util.get_project_path(project)
    ds = util.load_data_set(project_id)

    fs = project["features"].copy()
    plot_files = []
    
    count = 7

    for f1 in project["features"]:
        fs.remove(f1)
        print('>>>' + str(fs))
        for f2 in fs:
            count = count - 1
            if count <= 0:
                break
            print(f1 + '-' + f2)
            g = sns.FacetGrid(ds, hue =project["label"][0],
                height = 6)
            g.map(plt.scatter,f1,f2)
            g.add_legend()
            plot_file_name =  f1 +'-'+f2+'.png'
            g.savefig(project_path + plot_file_name)
            plot_files.append(plot_file_name)

    return plot_files

###############################################################################
# Plot v2
###############################################################################
@app.get("/projects/plot/v2/{project_id}")
def plot_data(project_id: int, algorithm: str):
    logger.info("plot_model project_id=", str(project_id), " algorithm=", algorithm)
    # http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/#example-2-decision-regions-in-1d
    project = util.get_project_by_id(project_id)
    project_path = util.get_project_path(project)
    dataset = util.load_data_set(project_id)
    dataArray = dataset.values
    print('==================', dataArray)


    fs = project["features"].copy()
    plot_files = []
    
    count = 7

    for f1 in project["features"]:
        fs.remove(f1)
        print('>>>' + str(fs))
        for f2 in fs:
            count = count - 1
            if count <= 0:
                break
            print(f1 + '-' + f2)

            model = util.load_model(project_id, 'KNeighborsClassifier')
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
            util.store_project_list()

            #train on a single model
            # Split-out validation dataset
            array = dataset.values
            X = array[0:,0:len(features)]
            y = array[0:,len(features)]
            print("X>>>>>>>>>>>>>>>>>>>>>>>" + str(X))
            print("y>>>>>>>>>>>>>>>>>>>>>>>" + str(y))
            model.fit(X, y)
            yNew =  np.array([])
            for value in y:
                if value =='Iris-setosa':
                    yNew = np.append(yNew, [int(0)])
                if value =='Iris-versicolor':
                    yNew = np.append(yNew, [int(1)])
                if value =='Iris-virginica':
                    yNew = np.append(yNew, [int(2)])
            yNew = yNew.astype(int)
            XNew = X.astype(float)
            print("yNew>>>>>>>>>>>>>>>>>>>>>>>" + str(yNew))
                                    
            fig = plot_decision_regions(X=XNew, y=yNew, clf=model, legend=2)
            plt.xlabel(f1)
            plt.ylabel(f2)
            plt.title('Knn with K=')
            plt.show()            
            
            plot_file_name =  f1 +'-'+f2+'.png'
            plt.savefig(project_path + plot_file_name)
            plot_files.append(plot_file_name)

    return plot_files


###############################################################################
# get plot file
###############################################################################
@app.get("/projects/plot_file/{project_id}", response_class=FileResponse)
def get_plot_file(project_id: int, plot_file_name: str):
    logger.info("get_plot_file project_id=", str(project_id), " plot_file_name=", plot_file_name)
    project = util.get_project_by_id(project_id)
    project_path = util.get_project_path(project)
    plot_file_path = project_path + plot_file_name
    base64_file_path = plot_file_path.replace("png","txt")

    util.png_to_base64(project_id, plot_file_path, base64_file_path)

    return base64_file_path
   
###############################################################################
# get algorithms
###############################################################################
@app.get("/algorithms")
def get_algorithms():
    print("get_algorithms")
    return app.algorithms
     
#
# Exceptions
#

@app.exception_handler(project.NotFoundException)
async def NotFoundException_exception_handler(request: Request, exc: project.NotFoundException):
    return JSONResponse(
        status_code=401,
        content={"message": f"Oops! {exc.message} did something. There goes a rainbow..."},
    )  