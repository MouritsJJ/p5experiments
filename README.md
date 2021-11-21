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
python=3.9.0\
absl-py==1.0.0\
appnope==0.1.2\
astunparse==1.6.3\
backcall==0.2.0\
beautifulsoup4==4.10.0\
bs4==0.0.1\
cachetools==4.2.4\
certifi==2021.10.8\
charset-normalizer==2.0.7\
click==8.0.3\
configparser==5.1.0\
cycler==0.11.0\
decorator==5.1.0\
docker-pycreds==0.4.0\
flatbuffers==2.0\
gast==0.4.0\
gitdb==4.0.9\
GitPython==3.1.24\
google-auth==2.3.3\
google-auth-oauthlib==0.4.6\
google-pasta==0.2.0\
grpcio==1.42.0\
h5py==3.6.0\
idna==3.3\
imageio==2.11.1\
importlib-metadata==4.8.2\
ipython==7.29.0\
jedi==0.18.1\
keras==2.7.0\
Keras-Preprocessing==1.1.2\
kiwisolver==1.3.2\
libclang==12.0.0\
Markdown==3.3.6\
matplotlib==3.4.3\
matplotlib-inline==0.1.3\
numpy==1.21.3\
oauthlib==3.1.1\
opt-einsum==3.3.0\
pandas==1.3.4\
parso==0.8.2\
pathtools==0.1.2\
pexpect==4.8.0\
pickleshare==0.7.5\
Pillow==8.4.0\
promise==2.3\
prompt-toolkit==3.0.22\
protobuf==3.19.1\
psutil==5.8.0\
ptyprocess==0.7.0\
pyasn1==0.4.8\
pyasn1-modules==0.2.8\
Pygments==2.10.0\
pyparsing==3.0.4\
python-dateutil==2.8.2\
pytz==2021.3\
PyYAML==6.0\
requests==2.26.0\
requests-oauthlib==1.3.0\
rsa==4.7.2\
sentry-sdk==1.4.3\
shortuuid==1.0.1\
six==1.16.0\
smmap==5.0.0\
soupsieve==2.2.1\
subprocess32==3.5.4\
tensorboard==2.7.0\
tensorboard-data-server==0.6.1\
tensorboard-plugin-wit==1.8.0\
tensorflow==2.7.0\
tensorflow-estimator==2.7.0\
tensorflow-io-gcs-filesystem==0.22.0\
termcolor==1.1.0\
torch==1.10.0\
torchvision==0.11.1\
traitlets==5.1.1\
typing-extensions==3.10.0.2\
urllib3==1.26.7\
wandb==0.12.6\
wcwidth==0.2.5\
Werkzeug==2.0.2\
wrapt==1.13.3\
yaspin==2.1.0\
zipp==3.6.0
