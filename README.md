# AuxiliaryTasks
For CMPUT701 Capstone on Auxiliary Tasks, at the University of Alberta, Fall 2018

#####Description#####

A research project that explores the impact of auxiliary tasks, as defined in Jaderberg, Max, et al. "Reinforcement learning with unsupervised auxiliary tasks." arXiv preprint arXiv:1611.05397 (2016).,
in a simple gridworld environment to observe their impact on environments with different reward and state transition dynamics

####Installation####

NOTE: This is provided as a script 'get_ubuntu_deps.sh' in the Scripts folder and was written and tested on Ubuntu 16.04

#Install GCC and G++ compilers
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-7 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc
gcc --version
g++ --version

#Install the Blas numerical computing library
sudo apt install libblas-dev -y

#Binaries for graph visualization used by Keras
sudo apt-get install graphviz -y

#Install virtualenv
sudo apt install virtualenv

#Create virtual environment and navigate to it, and activate it
virtualenv AUX
cd AUX
source bin/activate

#Clone the repo, enter it, and install python dependencies
git clone https://github.com/Rosevear/AuxiliaryTasks.git
cd Auxiliary-Tasks
python pip install -r requirements.txt

#####TESTING#####
#Run the unit tests for the environment

Navigate to the tests folder and run each individual test file from the command line (e.g): python test_grid_env.py

#TODO: Add tests for experiment and agent files

#####RUNNING#####
#Run the experiments for a particular parameter setting 

python grid_exp.py ARG_OPTIONS (look in grid_exp.py at the arg parser or use the command line to learn about the options)

OR

See the various bash scripts to run across varous environments and sweep the hyperparameters (the parameters swept over can be modified in grid_exp.py)

