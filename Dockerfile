FROM ros:foxy-ros-base

# Set the working directory
WORKDIR /workspace

# Install ROS 2 and Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-colcon-common-extensions \
    ros-foxy-rclpy \
    ros-foxy-std-msgs \
    ros-foxy-geometry-msgs \
    ros-foxy-sensor-msgs \
    ros-foxy-nav-msgs \
    ros-foxy-launch \
    ros-foxy-launch-ros \
    ros-foxy-rviz2 \
    ros-foxy-ament-cmake-python \
    python3-pytest \
    && rm -rf /var/lib/apt/lists/*

# Copy the package files into the container
COPY . .

# Create a minimal CMakeLists.txt to satisfy colcon
RUN echo 'cmake_minimum_required(VERSION 3.5)' > CMakeLists.txt && \
    echo 'project(autodrive_f1tenth)' >> CMakeLists.txt && \
    echo 'find_package(ament_cmake REQUIRED)' >> CMakeLists.txt && \
    echo 'ament_package()' >> CMakeLists.txt

# Install all the Python dependencies from the README
RUN python3 -m pip install \
    eventlet==0.33.3 \
    Flask==1.1.1 \
    Flask-SocketIO==4.1.0 \
    python-socketio==4.2.0 \
    python-engineio==3.13.0 \
    greenlet==1.0.0 \
    gevent==21.1.2 \
    gevent-websocket==0.10.1 \
    Jinja2==3.0.3 \
    itsdangerous==2.0.1 \
    werkzeug==2.0.3 \
    attrdict \
    numpy \
    pillow \
    opencv-contrib-python \
    pynput \
    flake8

# Fix setuptools and importlib_metadata compatibility issue
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install setuptools==59.6.0 importlib_metadata==4.8.3 wheel

# Install the package in development mode
RUN . /opt/ros/foxy/setup.sh && \
    python3 -m pip install -e src/autodrive_f1tenth

# Build the ROS 2 workspace with detailed output
# RUN . /opt/ros/foxy/setup.sh && \
#     colcon build --packages-select autodrive_f1tenth --event-handlers console_direct+

# Source the ROS 2 setup script and workspace
CMD ["bash", "-c", "source /opt/ros/foxy/setup.sh && source install/setup.bash && exec bash"]