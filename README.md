# Purpose
Code base for documenting experiments conducted during a 5th semester project at Aalborg University (AAU). The project is experimenting on how to generate images of cars having certain features. The project concerns the theme: Synthetic Data Generation, as it tries to generate synthetic product images that should be used to better display products that do not have enough images for showcasing. 

# Running the experiments
Each folder is related to an experiment conducted in the project. This is with the exception of a few folders which contain the used images for training the models. 

To run an experiment change your current directory to a folder containing a `main.py`-file, which should be run using python. For common systems the command should look like:\
`python main.py` or\
`python3 main.py`.

# Notes
The experiments has been logged using Weight and Biases, which can be found as their website `wandb.ai`. To log graphs and other data according to the presented results in the project report it is necessary to create a wandb account and change any line in the `main.py`-files concerning wandb logging.

Example:\
`wandb.init(project='<project name>', entity='<Entity name>', name='<Name of run>', notes='<Notes>')`

If the line has been changed to use your own wandb account, the command for running the experiments will be:\
`python main.py -w 1` or\
`python3 main.py -w 1`.

# Dependencies with versions
The project uses python 3.9.0.\
The dependencies with versions can be found in requirements.txt.