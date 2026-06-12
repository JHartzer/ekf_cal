FROM osrf/ros:jazzy-desktop

# non interactive frontend for locales
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get -y install \
    build-essential \
    clang-15 \
    cloc \
    cmake \
    doxygen \
    g++ \
    gdb \
    git \
    lcov \
    libeigen3-dev \
    libgtest-dev \
    libhdf5-dev \
    libopencv-dev libopencv-contrib-dev \
    libyaml-cpp-dev \
    linux-tools-generic \
    locales \
    python3 python3-pip python3-venv \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set Locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Source ROS2
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

# Install Aprilgrid
RUN mkdir -p /ekf_cal_ws/src && \
    cd /ekf_cal_ws/src && \
    git clone https://github.com/JHartzer/aprilgrid.git && \
    mkdir /ekf_cal_ws/src/aprilgrid/build -p && \
    cd /ekf_cal_ws/src/aprilgrid/build && \
    cmake .. && \
    cmake --build . && \
    make install
