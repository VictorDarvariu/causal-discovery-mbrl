"""
This script was integrated without modification from the RL-BIC repository. Original contents follow.
"""
import rpy2.robjects.packages as rpackages
from rpy2 import robjects as robj

utils = rpackages.importr('utils')
base = rpackages.importr('base')

pck_names = ['Matrix', 'mgcv', 'codetools', 'rpart', 'glmnet', 'mboost', 'CAM']

print(f"Installing to R lib path <<<{base._libPaths()}>>>")

for pck_name in pck_names:
    if rpackages.isinstalled(pck_name):
        continue
    if pck_name != 'CAM':
        utils.install_packages(pck_name, lib="/usr/lib/R/library")
    else:
        #install CAM locally
        utils.install_packages('CAM_1.0.tar.gz', repos=robj.rinterface.NULL, type='source', lib="/usr/lib/R/library")
        print('R packages CAM has been successfully installed.')

# check if CAM and mboost have been installed
if rpackages.isinstalled('CAM', lib_loc="/usr/lib/R/library") and rpackages.isinstalled('mboost', lib_loc="/usr/lib/R/library"):
    print('R packages CAM and mboost have been installed')
else:
    print('need to install CAM and mboost')

