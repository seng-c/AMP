#!/bin/bash
conda activate amp
python3 ./main_detector.py train/ main_train.json
python3 ./template/evaluate.py train/ main_train.json >> evaluation_results.txt
echo "---------------------------------------" >> evaluation_results.txt
echo "" >> evaluation_results.txt
conda deactivate
clear