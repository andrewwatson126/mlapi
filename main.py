from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from pandas import read_csv
from matplotlib import pyplot

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

models = [ 
        ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')), 
        ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
        ('KNeighborsClassifier', KNeighborsClassifier()),
        ('DecisionTreeClassifier', DecisionTreeClassifier()),
        ('GaussianNB', GaussianNB()),
        ('SVC', SVC(gamma='auto'))
    ]


class Project(BaseModel):
    id: int
    name: str    
    created_date: datetime
    description: Optional[str]
    experiments: Optional[List[int]] = None


class Experiment(BaseModel):
    id: int
    name: str    
    created_date: datetime
    description: Optional[str]
    fileUrl: str
    features: List[int]
    label: str

experiment = {   
        "id": 1, 
        "name": "experiment-1", 
        "created_date":"2021-06-11 12:00", 
        "description":"experiment 1 desc", 
        "fileUrl": "iris.csv",
        "algorithms" : ["LogisticRegression", "LinearDiscriminantAnalysis", "KNeighborsClassifier", "DecisionTreeClassifier", "GaussianNB", "SVC"],
        "accuracy" :  {},
        "model_dict": {}
    }


project_list =[
        #{ Project(1, "projet-1", "desc 1",  11) }, 
        #{ Project(id=2, name="projet-2", description="desc 2", experienceId=22)},
        {   "id": 1, 
            "name": "projet-1", 
            "created_date":"2021-06-11 12:00", 
            "description":"desc 1", 
            "experienceId": 11,
            "experiments" : [11, 12, 13]
        },
        {   "id": 2, 
            "name": "projet-2", 
            "created_date":"2021-06-11 12:00", 
            "description":"desc 2", 
            "experienceId": 22,
            "experiments" : [21, 22, 23]
        },
        {   "id": 3, 
            "name": "projet-3", 
            "created_date":"2021-06-11 12:00", 
            "description":"desc 3", 
            "experienceId": 33,
            "experiments" : [31, 32, 33]
        }
]

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/test")
def get_test():
    return {"message": "Hello World"}


@app.get("/projects")
def get_project_list():
    return project_list


@app.get("/projects/{project_id}")
def get_project(project_id: int):
    for project in project_list:
        if project.get("id") == project_id:
            return project

@app.post("/projects")
async def create_project(project: Project):
    logger.info("logging from the root logger")
    project_list.append(project)
    return project

@app.put("/projects/{project_id}")
async def update_project(project_id: int, project: Project):
    project_list.update(project)
    return project

@app.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    for project in project_list:
        if project.get("id") == project_id:
            project_list.remove(project)

@app.get("/experiments/loadFile")
def load_file(experiment_id: int):
    # read file - filename provided from the API
    # set the filename in the experiment
    dataset = read_csv(experiment.get('fileUrl'))

    # set supervised/unsupervised - get supervise / unsupervised from the API
    #se the supervised flag to TRUE in the experiment

    # set the features to from 0:n-2 and label to n-1
    names = list(dataset.columns)
    features = []
    features = names[0: len(names)-1]
    labels = []
    labels = names[len(names)-1: len(names)]
    label = labels[0]

    #train on a single model
    # Split-out validation dataset
    array = dataset.values
    X = array[1:,0:4]
    y = array[1:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

    # check algorithms
    # evaluate each model in turn
    results = []
    xnames = []
    accuracyDict = {}
    model_dict = {}
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        xnames.append(name)
        #accuracy['LogisticRegression'] = [cv_results.mean(), cv_results.std()]
        accuracyDict[name] =  cv_results.mean()

        # create model
        m = model.fit(X_train,Y_train)
        model_dict[name] = m
        model_dict[name].predict([[5.1, 3.5, 1.4, 0.2]])
        #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))    

    experiment["model_dict"] = model_dict
    return {"features": features},{"label": label}, {"accuracy": accuracyDict}

@app.post("/experiments/predict/{experiment_id}")
def predict(experiment_id: int, data: list ):
    logger.info("predict(experiment_id=" + str(experiment_id) + ")")
    predictDict = {}
    model_dict = experiment["model_dict"] 
    logger.info("model_dict=" + str(model_dict))

    for name in model_dict:
        logger.info("name=" + name)
        d = data 
        #d = [[6.1,2.9,4.7,1.4]]
        logger.info("name=" + name + "=" +  str(model_dict[name].predict(d)))
        predictDict[name] = str(model_dict[name].predict(d))
        #predictDict.update({name: model_dict[name].predict(d)})

    return {"predictDict": predictDict }




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

