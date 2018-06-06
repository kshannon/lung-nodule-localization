#!/bin/bash
echo "I am combining all of the subset predictions into one called predictions_ALL.csv"
cp predictions_subset0.csv predictions_ALL.csv

for i in `seq 1 9`;
do
  tail -n+2 predictions_subset$i.csv >> predictions_ALL.csv
done

echo "I am updating x,y,z centers to shift for 64 pixels"
python correct_centers.py
echo "Done. Now run evaluationScript/noduleCADEvalutions.py."
