FROM osrf/ros:jazzy-desktop

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# non interactive frontend for locales
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get -y install \
    build-essential \
    clang clangd \
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
    ninja-build \
    python3 python3-pip python3-venv \
    sudo \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN existing_group="$(getent group "${USER_GID}" | cut -d: -f1 || true)" && \
    existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)" && \
    if [ -z "${existing_group}" ]; then \
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    elif [ "${existing_group}" != "${USERNAME}" ] && ! getent group "${USERNAME}" >/dev/null; then \
    groupmod --new-name "${USERNAME}" "${existing_group}"; \
    fi && \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
    usermod --uid "${USER_UID}" --gid "${USER_GID}" "${USERNAME}"; \
    elif [ -n "${existing_user}" ]; then \
    usermod --login "${USERNAME}" --home "/home/${USERNAME}" --move-home --gid "${USER_GID}" "${existing_user}"; \
    else \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}"; \
    fi && \
    mkdir -p /home/${USERNAME} && \
    chown -R "${USER_UID}:${USER_GID}" /home/${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Install python packages
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# Set Locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Source ROS2
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/${USERNAME}/.bashrc

# Install Aprilgrid
RUN mkdir -p /ekf_cal_ws/src && \
    cd /ekf_cal_ws/src && \
    git clone https://github.com/JHartzer/aprilgrid.git && \
    mkdir /ekf_cal_ws/src/aprilgrid/build -p && \
    cd /ekf_cal_ws/src/aprilgrid/build && \
    cmake .. && \
    cmake --build . && \
    make install && \
    chown -R ${USERNAME}:${USERNAME} /ekf_cal_ws /home/${USERNAME}
