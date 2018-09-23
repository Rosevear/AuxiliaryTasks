#####Description#####
A research project that explores the impact of auxiliary tasks, as defined in Jaderberg, Max, et al. "Reinforcement learning with unsupervised auxiliary tasks." arXiv preprint arXiv:1611.05397 (2016).

####Installation####
#Make sure you have python (2.7.12) and the pip package manager installed

#Install virtualenv: https://virtualenv.pypa.io/en/stable/ (it is used to create a virtual environment that keeps python dependencies separate)

#Create a virtualenv to house the project
virtualenv --python=PATH_TO_PYTHON project_name

#Enter the environemtn
cd project_name
source bin/activate

#Get the repository
git clone https://github.com/Rosevear/AuxiliaryTasks.git

#Install the dependencies via pip
cd AuxiliaryTasks
pip install -r requirements.txt


#Run the unit tests for the environment
python test_grid_env.py
#TODO: Add tests for experiment and agent files

#Run the experiments for a particular parameter setting 
python grid_exp.py ARG_OPTIONS (look in grid_exp.py at the arg parser or use the command line to learn about the options)

OR
#Run the experiments across all agent environments and sweep the parameter settings
source run_experiments.sh
