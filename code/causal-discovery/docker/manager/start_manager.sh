#! /bin/bash
jupyter notebook --no-browser --notebook-dir=/causal-discovery --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --allow-root &
tail -f /dev/null