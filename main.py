from typing import Optional

from fastapi import FastAPI, Request
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


algorithms = [
    { 
     "name": "Logistic Regression", 
     "label": "LogisticRegression" 
     },
    { 
     "name": "Linear Discriminant Analysis", 
     "label": "LinearDiscriminantAnalysis" 
     },
    { 
     "name": "KNeighbors Classifier", 
     "label": "KNeighborsClassifier" 
     },
    { 
     "name": "Decision Tree Classifier", 
     "label": "DecisionTreeClassifier" 
     },
    { 
     "name": "Gaussian NB", 
     "label": "GaussianNB" 
     },
    { 
     "name": "SVC", 
     "label": "SVC" 
     }
]


#class Model(Enum):
#    SUPERVISED = 'Supervised'
#    UNSUPERVISED = 'Unsupervised'

class Project(BaseModel):
    id: int = 0
    name: str = ""  
    created_date: Optional[datetime] 
    description: Optional[str] = ""
    data_file: Optional[str] = ""
    createdBy: str = ""
    model: str = "Supervised"
    algorithms: List[str] = []
    features: List[str] = []
    label: List[str] = []

project_data = {
    "id": 1,
    "name": "diagnostic1",
    "created_date": "2021-12-21 00:00",
    "description": "1aim is to diagnose for abc",
    "data_file": "1abc.xlsx",
    "createdBy": "nevilgultekin",
    "model": "Supervised",
    "algorithms": [ "LogisticRegression" ],
    "features": [ "age", "gender", "test2", "test3"],
    "label": [ "diagnose"]
}

#project = Project(**project_data)




#@app.get("/")
#def read_root():
#    return FileResponse("index.html")

#@app.get("/test")
#def get_test():
#    return {"message": "Hello World"}

app.project_list =[]


# get project list
@app.get("/projects", response_model=List[Project])
def get_project_list():
    logger.info("1-get_project_list", app.project_list)
    
    readProjectList()
    
    logger.info("2-get_project_list=", app.project_list)
    return app.project_list

# get project 
@app.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: int):
    logger.info("get_project")
    return get_project_by_id(project_id)

def get_project_by_id(project_id: int):
    logger.info("get_project_by_id")
    readProjectList()
    for project in app.project_list:
        if project.get("id") == project_id:
            return project
        
    raise NotFoundException("Project with id , project_id,  not found")

# create project 
@app.post("/projects")
async def create_project(project: Project):
    logger.info("create_project=", project)

    readProjectList()

    exists = False
    for p in app.project_list:
        if p["id"] == project.id:
            exists = True
    if exists == False:
        project_json = jsonable_encoder(project)
        app.project_list.append(project_json)
    logger.info("create_project appended=", app.project_list)
        
    storeProjectList()
    readProjectList()

    return project

# update project 
@app.put("/projects/{project_id}")
async def update_project(project_id: int, project: Project):
    logger.info("update_project")
    p = get_project_by_id(project.id)
    p["name"] = project.name
    p["created_date"] = project.created_date
    p["description"] = project.description
    p["data_file"] = project.data_file
    p["createdBy"] = project.createdBy
    p["model"] = project.model
    p["algorithms"] = project.algorithms
    p["features"] = project.features
    p["label"] = project.label

    storeProjectList()
    readProjectList()

    return project

@app.delete("/projects/{project_id}")
async def delete_project(project_id: int):

    readProjectList()

    for project in app.project_list:
        if project.get("id") == project_id:
            app.project_list.remove(project)
            
    storeProjectList()
    readProjectList()
            

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

# Store and Read Project List 
def storeProjectList():
    logger.info("***storeProjectList=",app.project_list)
    with open('project_list.json', 'w') as f:
        json.dump(app.project_list, f, ensure_ascii=False, indent=4)
    f.close()

def readProjectList():
    f = open('project_list.json')
    app.project_list = json.load(f)
    logger.info("readProjectList")
    logger.info(app.project_list)
    f.close()

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