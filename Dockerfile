FROM python:3.8-slim-buster

WORKDIR /offlax

COPY . .
RUN python -m pip install .

RUN mkdir /workspace
WORKDIR /workspace
