#!/bin/bash
expPart=$1
expWhich=$2
expIdSuffix=$3
expGt=$4
expBudget=$5
expVary=$6


if [[ $expVary == "density" ]];
then
  Ns=(1000)
  P=10
  Es=(15 20) # 25 30 35 40 45)
else
  Ns=(10 25) # 50 75 100 175 250 375 500 750 1000)
  P=10
  Es=(30)
fi

deploymentHost=$(echo $HOSTNAME | cut -d "." -f 1)


for i in ${!Es[@]};
do
  E=${Es[$i]}

  for j in ${!Ns[@]};
  do
    N=${Ns[$j]}

    if [[ $expVary == "density" ]];
    then
      fullSuffix=${expIdSuffix}_${expGt}_e${E}
    else
      fullSuffix=${expIdSuffix}_${expGt}_n${N}
    fi

    instanceName="synth${P}gpr"
    expId=${instanceName}_${fullSuffix}


docker exec -it cd-manager /bin/bash -c "source activate cd-env && python /causal-discovery/cdrl/setup_experiments.py \
                --experiment_id ${expId} --which ${expWhich} --experiment_part ${expPart} --instance_name ${instanceName} \
                --budget ${expBudget} \
                --gt ${expGt} --n ${N} --p ${P} --e ${E} \
                --what_vary ${expVary}"

    taskCount=$(cat $CD_EXPERIMENT_DATA_DIR/${expId}/models/${expPart}_tasks.count | tr -d '\n')

    for i in $(seq 1 $taskCount); do
      (docker exec -d cd-manager /bin/bash -c "source activate cd-env && python /causal-discovery/cdrl/tasks.py --experiment_id ${expId} --experiment_part ${expPart} --task_id $i") &
    done

  done

done




