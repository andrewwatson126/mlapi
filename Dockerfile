FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED 1 

EXPOSE 8000

WORKDIR /app

# Copy requirements from host, to docker container in /app 
COPY ./requirements.txt .

# Copy everything from ./src directory to /app in the container
COPY ./src .
COPY ./data ./data

RUN pip install -r requirements.txt

# Run the application in the port 8000
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]