FROM ubuntu:latest

RUN apt-get update &&\
    apt-get install -y apt-utils &&\
    apt-get install -y curl &&\
    apt-get install ffmpeg libsm6 libxext6  -y &&\
    apt-get install -y python3 &&\
    apt-get install -y python3-pip &&\
    apt-get install libglib2.0-0 &&\
    apt-get install -y git


# Move our code over 
ADD ./ ~/../usr/code 

# Install our requirements
RUN pip3 install -r ~/../usr/code/env/requirements.txt 

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH"

# Install CGAL
RUN apt-get install -y libcgal-dev &&\
    apt-get install -y swig &&\
    apt-get install -y build-essential libssl-dev 

RUN cd ../tmp &&\
    apt-get install -y wget &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz &&\
    tar -zxvf cmake-3.20.0.tar.gz &&\
    cd cmake-3.20.0 &&\
    ./bootstrap &&\
    make &&\
    make install &&\
    cmake --version 

RUN cd .. &&\
    git clone https://github.com/cgal/cgal-swig-bindings &&\
    cd cgal-swig-bindings &&\
    mkdir build/CGAL-5.0_release -p &&\
    cd build/CGAL-5.0_release &&\
    cmake -DCGAL_DIR=/usr/lib/CGAL -DBUILD_JAVA=OFF -DPYTHON_OUTDIR_PREFIX=../../examples/python ../.. &&\
    make -j 4 &&\
    cd ../../examples/python &&\ 
    cp -r CGAL ~/../usr/code/src/




# CMD [ "python3" , "./usr/code/main.py"]