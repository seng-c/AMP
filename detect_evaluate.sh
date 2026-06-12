#!/bin/bash
python3 ./main_detector.py train/ main_train.json
python3 ./template/evaluate.py train/ main_train.json
