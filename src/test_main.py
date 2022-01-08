from fastapi.testclient import TestClient
from matplotlib.pyplot import plot
from sklearn.metrics.pairwise import polynomial_kernel
from .main import app

client = TestClient(app)


def test_get_projects():
    response = client.get("/projects")
    assert response.status_code == 200
    #assert response.json() == {"msg": "Hello World"}

def test_create_project():
    response = client.post(
        "/projects",
        json={
            "id": 0,
            "name": "unittestname",
            "created_date": "2022-01-08T16:15:46.508Z",
            "description": "unittestdesc",
            "data_file": "",
            "created_by": "unittestuser",
            "model": "Supervised",
            "algorithms": [],
            "features": [],
            "label": [],
            "accuracy": {}
        },
    )
    assert response.status_code == 200
    #assert response.json() == {
    #    "name": "unittestname",
    #}

#def test_upload_file():
#    response = client.post("/projects/uploadfile?project_id=1")
#    assert response.status_code == 200
#    #assert response.json() == {"msg": "Hello World"}


def test_get_project():
    response = client.get("/projects/1")
    assert response.status_code == 200
    #assert response.json() == {"msg": "Hello World"}


#def test_update_project():
#    response = client.put("/projects",
#        json={
#            "id": 1,
#            "name": "unittestname",
#            "created_date": "2022-01-08T16:15:46.508Z",
#            "description": "unittestdesc",
#            "data_file": "",
#            "created_by": "unittestuser",
#            "model": "Supervised",
#            "algorithms": [],
#            "features": [],
#            "label": [],
#            "accuracy": {}
#        },
#    )
#    assert response.status_code == 200
#    #assert response.json() == {
#    #    "name": "unittestname",
#    #}

#def test_get_correlation_values():
#    response = client.get("/projects/correlation_values/1")
#    assert response.status_code == 200
#    #assert response.json() == {"msg": "Hello World"}

def test_get_features_labels():
    response = client.get("/projects/features_labels/1")
    assert response.status_code == 200
    #assert response.json() == {"msg": "Hello World"}

def get_correlation_plot():
    response = client.get("/projects/correlation/1")
    assert response.status_code == 200
    #assert response.json() == {"msg": "Hello World"}
    
def get_plot_model():
    response = client.get("/projects/plot/1")
    assert response.status_code == 200
    #assert response.json() == {"msg": "Hello World"}
    
def get_algorithms():
    response = client.get("/algorithms")
    assert response.status_code == 200
    #assert response.json() == {"msg": "Hello World"}