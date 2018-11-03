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

#Install virtualenv
sudo apt install virtualenv

#Create virtual environment and navigate to it, and activate it
virtualenv AUX
cd AUX
source bin/activate

#Clone the repo, enter it, and install python dependencies
git clone https://github.com/Rosevear/AuxiliaryTasks.git
cd AuxiliaryTasks
pip install -r requirements.txt
