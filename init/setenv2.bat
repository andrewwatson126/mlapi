set PATH=%PATH%;C:\Users\nevil\AppData\Local\Programs\Python\Python310\Scripts
set PATH=C:\Python310;%PATH%;

REM install python
REM install nodejs

	set PATH=C:\Users\nevil\AppData\Local\Programs\Python\Python39;%PATH%
	set PATH=C:\Users\nevil\AppData\Local\Programs\Python\Python39\Scripts;%PATH%
	set PATH=C:\Users\nevil\AppData\Roaming\Python\Python39\Scripts;%PATH%
	set PATH=C:\Program Files\nodejs;%PATH%
	set PATH=C:\Program Files\git;%PATH%
	set PATH=C:\Users\nevil\bin\yum-3.4.3\bin;%PATH%

pip install Flask
pip install create-react-app
pip install fastapi[all]
pip install uvicorn
pip install starlette


pip install numpy
pip install pandas
pip install matplotlib
pip install StringGenerator
pip install sklearn
 pip install pickle-mixin

 npm uninstall -g create-react-app

npm install axios
npm install @mui/material @emotion/react @emotion/styled
npm install @material-ui/core
npm i @material-ui/core


create-react-app <app-name>
npm start

cd C:\Users\nevil\PycharmProjects\fast1
uvicorn main:app --reload
uvicorn main:app --reload --host 192.168.1.15 --log-config log.ini
uvicorn src.main:app --reload --host 192.168.1.49
uvicorn main:app --log-config log.ini --reload


http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc


python -m pip install --upgrade pip


pip install -U Flask
pip install -U create-react-app
pip install -U fastapi[all]
pip install -U uvicorn
pip install -U starlette
pip install -U numpy
pip install -U pandas
pip install -U matplotlib
pip install -U StringGenerator
pip install -U scipy
pip install -U joblib
pip install -U threadpoolctl
pip install -U cython
pip install -U scikit-image
pip install -U seaborn
pip install -U memory_profiler
pip install -U fastapi
pip install -U sklearn

npm install @mui/icons-material
npm install @mui/material @mui/styled-engine-sc styled-component
npm install @emotion/react @emotion/styled
npm install recharts
npm install @mui/styles

npm install @material-ui/core
npm install @material-ui/icons
npm install axios
npm install react-router-dom
npm install axios
npm install js-file-download


Inside that directory, you can run several commands:

  npm start
    Starts the development server.

  npm run build
    Bundles the app into static files for production.

  npm test
    Starts the test runner.

  npm run eject
    Removes this tool and copies build dependencies, configuration files
    and scripts into the app directory. If you do this, you can’t go back!

We suggest that you begin by typing:

  cd mlstudio2
  npm start
  
  
  
 Material UI notes
 * makestyles
 * Grid container -> Grid item
 
 
 
 # useful links
 List and dashboard - https://www.youtube.com/watch?v=CNh3Q_z9GSw
 Form example - https://www.youtube.com/watch?v=5akdtwtmjZM - React-Material-UI-Dialog-Modal-Popup
 
 
 
 NGINX
 =====================================
 cd c:\
unzip nginx-1.2.3.zip
ren nginx-1.2.3 nginx
cd nginx
start nginx

nginx -s [ stop | quit | reopen | reload ]