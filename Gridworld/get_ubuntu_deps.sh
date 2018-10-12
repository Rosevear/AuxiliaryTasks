#Binaries for graph visualization used by Keras
sudo apt-get install graphviz -y

#GCC and G++ compilers
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-7 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc
gcc --version
g++ --version

#Blas numerical computing library
sudo apt install libblas-dev -y
