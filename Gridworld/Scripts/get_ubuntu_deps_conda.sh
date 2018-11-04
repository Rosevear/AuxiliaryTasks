#!/bin/bash
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

#Get Anaconda
wget https://repo.anaconda.com/archive/Anaconda2-5.3.0-Linux-x86_64.sh

#Install if via bash script
bash ~/Downloads/Anaconda2-5.3.0-Linux-x86_64.sh

#Create a virtual environment in which to house the project
conda create --name AUX python=2.7

#Theano dependencies (last three are optional)
conda install numpy scipy mkl nose sphinx pydot-ng
pip install parameterized #(optional for uint testing)

#Install tehano and pygpu array
conda install theano pygpu

#Miscellaneous dependencies (mostly for graphing)
pip install matplotlib==1.5.3
pip install graphviz==0.8.2
pip install pydot

#Get Keras high level api
pip install keras
