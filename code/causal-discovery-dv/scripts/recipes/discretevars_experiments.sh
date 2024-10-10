#! /bin/bash

# Experiments on datasets with discrete variables (Table 2)
$CDDV_SOURCE_DIR/scripts/run_experiments.sh eval main discretevars 1000 asia
$CDDV_SOURCE_DIR/scripts/run_experiments.sh eval main discretevars 1000 insurance
$CDDV_SOURCE_DIR/scripts/run_experiments.sh eval main discretevars 1000 child
