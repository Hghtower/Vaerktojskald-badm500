Overview of files and data:

- train.py
The train.py file contains the training loop.

- validate.py 
validate.py contains the the code for testing the model with the test dataset, making the bar plots and giving accuracies. 

- generation.py 
generation.py is the script which has been used for generating dateset v2 and v3. 

- graphing.py
graphing.py has been used to create the 4 graphs for fine-tuning of the hyperparameters.

- plotmaster.py 
Script used for creating the plot accuracy and loss plot from results.

- validate_tools.py
Script for checking the validity of the datasets.

- data_processing.py
Script for removing duplicates of data and concatenate datasets of different models.

- Data folder 
The data_processed folder contains the second iteration of the dataset. The data_processed_v3 folder contains the third iteration of the dataset. 

- gemma-270m folder
Contains gemma checkpoints and tensorboard information

- graph data folder
Contains graph data for accuracy and loss curves 

- tools folder
Contains validate_tools.py



- Script for installing the virtual environment 
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt











