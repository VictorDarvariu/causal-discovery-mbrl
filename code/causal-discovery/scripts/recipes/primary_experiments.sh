#! /bin/bash

# Primary experiments on sachs and syntren (Table 1)
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 1178 sachs
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren1
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren2
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren3
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren4
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren5
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren6
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren7
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren8
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren9
$CD_SOURCE_DIR/scripts/run_experiments.sh hyperopt main primary 97 syntren10

# [await completion]
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 1178 sachs
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren1
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren2
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren3
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren4
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren5
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren6
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren7
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren8
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren9
$CD_SOURCE_DIR/scripts/run_experiments.sh eval main primary 97 syntren10