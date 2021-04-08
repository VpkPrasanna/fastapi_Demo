FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /app
WORKDIR /app

RUN apt-get update -y
RUN apt-get install virtualenv python3-pip -y
RUN pip install --upgrade pip
RUN pip install -r requirement.txt
