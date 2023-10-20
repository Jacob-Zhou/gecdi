#!/bin/bash

[ ! -d "venv-py36" ] && virtualenv -p python3.6 venv-py36 && . venv-py36/bin/activate && python -m pip install errant==2.0.0 && python -m spacy download en
. venv-py36/bin/activate
