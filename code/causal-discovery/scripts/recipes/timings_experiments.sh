#! /bin/bash
instances=("synth10lr" "synth15lr" "synth20lr" "synth25lr" "synth30lr" "synth35lr" "synth40lr" "synth45lr" "synth50lr")

for i in ${!instances[@]};
do
  instance=${instances[$i]}
  $CD_SOURCE_DIR/scripts/run_experiments.sh eval timings timings 10 ${instance}

done

echo "Done launching everything."