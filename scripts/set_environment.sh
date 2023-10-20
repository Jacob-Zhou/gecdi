#!/bin/bash

[ ! -d venv ] && virtualenv venv && . venv/bin/activate && pip install -r requirements.txt && python -m spacy download en
. venv/bin/activate
