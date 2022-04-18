#!/usr/bin/bash

git clone -b $RNN_BRANCH --single-branch $RNN_REPO --depth 1 ./rnn

cd ./rnn
pip install -r requirements.txt
python setup.py install

tune $TUNE_CONFIG