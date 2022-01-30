uvicorn  src.main:app --reload --port 8080 &
sleep 3
uvicorn  src.main:app --reload --port 8081 &
sleep 3
uvicorn  src.main:app --reload --port 8082 &
sleep 3
uvicorn  src.main:app --reload --port 8083 &
sleep 3
uvicorn  src.main:app --reload --port 8084 &
sleep 3
uvicorn  src.main:app --reload --port 8085 &
sleep 3
uvicorn  src.main:app --reload --port 8086 &
sleep 3
uvicorn  src.main:app --reload --port 8087 &
sleep 3

curl -X 'POST' 'http://localhost:8080/projects/best_model?project_id=1&start_from_index=1' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[ 1,9,17]' &
sleep 1
curl -X 'POST' 'http://localhost:8081/projects/best_model?project_id=1&start_from_index=2' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[ 2,10,18]' &
sleep 1
curl -X 'POST' 'http://localhost:8082/projects/best_model?project_id=1&start_from_index=3' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[ 3,11,19]' &
sleep 1
curl -X 'POST' 'http://localhost:8083/projects/best_model?project_id=1&start_from_index=4' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[ 4,12,20]' &
sleep 1
curl -X 'POST' 'http://localhost:8084/projects/best_model?project_id=1&start_from_index=5' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[ 5,13,21]' &
sleep 1
curl -X 'POST' 'http://localhost:8085/projects/best_model?project_id=1&start_from_index=6' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[ 6,14,22]' &
sleep 1
curl -X 'POST' 'http://localhost:8086/projects/best_model?project_id=1&start_from_index=7' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[ 7,15,23]' &
sleep 1
curl -X 'POST' 'http://localhost:8087/projects/best_model?project_id=1&start_from_index=8' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[ 8,16,24]' &
sleep 1


 cat best_model_higher_than_095_iteration_2801_1.json > best_model.json
 cat best_model_higher_than_095_iteration_2801_2.json >> best_model.json
 cat best_model_higher_than_095_iteration_2801_3.json >> best_model.json
 cat best_model_higher_than_095_iteration_2801_4.json >> best_model.json
 cat best_model_higher_than_095_iteration_2801_5.json >> best_model.json
 cat best_model_higher_than_095_iteration_2801_6.json >> best_model.json
 cat best_model_higher_than_095_iteration_2801_7.json >> best_model.json
 cat best_model_higher_than_095_iteration_2801_8.json >> best_model.json
