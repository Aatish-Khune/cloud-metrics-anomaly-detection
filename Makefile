PY ?= python

.PHONY: data train test app all

data:
	$(PY) run.py data

train:
	$(PY) run.py train

test:
	$(PY) run.py test

app:
	$(PY) run.py app

all:
	$(PY) run.py all
