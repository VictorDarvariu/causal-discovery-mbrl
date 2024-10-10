#! /bin/bash
expPart=$1
expWhich=$2
expIdSuffix=$3
expBudget=$4
instanceNames=($5)

echo "Running with experiment suffix $expIdSuffix."

for i in ${!instanceNames[@]};
do
  instanceName=${instanceNames[$i]}
  echo "Doing instance $instanceName"

  expId=${instanceName}_${expIdSuffix}

  docker exec -it cddv-manager /bin/bash -c "source activate cddv-env && python /causal-discovery-dv/cdrl/setup_experiments.py --experiment_part ${expPart} --which ${expWhich} --experiment_id ${expId} --instance_name ${instanceName} --budget ${expBudget}"
  taskCount=$(cat $CD_EXPERIMENT_DATA_DIR/${expId}/models/${expPart}_tasks.count | tr -d '\n')

  for i in $(seq 1 $taskCount); do
    docker exec -d cddv-manager /bin/bash -c "source activate cddv-env && python /causal-discovery-dv/cdrl/tasks.py --experiment_id ${expId} --experiment_part ${expPart} --task_id $i"
#    docker exec -it cddv-manager /bin/bash -c "source activate cddv-env && python /causal-discovery-dv/cdrl/tasks.py --experiment_id ${expId} --experiment_part ${expPart} --task_id $i"
  done

done

echo "Done launching everything."