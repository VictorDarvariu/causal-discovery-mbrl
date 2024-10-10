#! /bin/bash
expPart=$1

budgets=("1178" "500" "250" "100" "50" "25" "10" "5" "2.5" "1" "0.5" "0.25" "0.1")

for i in ${!budgets[@]};
do
  budget=${budgets[$i]}

  fullSuffix=varbudgetb${budget}
  $CD_SOURCE_DIR/scripts/run_experiments.sh ${expPart} budget ${fullSuffix} ${budget} sachs

done

echo "Done launching everything."
