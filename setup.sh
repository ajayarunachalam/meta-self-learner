#!/bin/sh

# assumes run from root directory

echo "Installing Meta Ensemble Self Learning Package" 
echo "START..."
pip install -r requirements.txt
echo "END"
xterm -e python -i -c "print('>>> from msl.MetaLearning import *');from msl.MetaLearning import *"
xterm -e python -i -c "print('>>> from msl.cf_matrix import make_confusion_matrix');from msl.cf_matrix import make_confusion_matrix"
echo "Test Environment Configured"