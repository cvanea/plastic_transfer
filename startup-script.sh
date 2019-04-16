#!/usr/bin/env bash

cd plastic_transfer
git pull origin master
pip install requirements.txt
python3 main_local.py
