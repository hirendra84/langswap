#SHELL := /bin/bash

CURRENT_USER := $(shell whoami)

# SET THIS! Directory containing wsgi.py
# PROJECT := someproject

LOCALPATH := ./joint
PYTHONPATH := $(LOCALPATH)/

LDFLAGS := "-L/usr/local/opt/openssl@1.1/lib"
CPPFLAGS := "-I/usr/local/opt/openssl@1.1/include"
LC_ALL := en_US.UTF-8

devel:
	pipenv --rm
	pipenv install --dev

run:
	pipenv run uvicorn api.main:app --reload

lock:
	pipenv --rm
	pipenv install
	pipenv lock -r > requirements.txt