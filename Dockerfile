FROM osrf/ros:humble-desktop

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

SHELL [ "/bin/bash" , "-c" ]
RUN apt-get update
RUN apt-get install -y git && apt-get install -y python3-pip && apt install -y python3-colcon-common-extensions
RUN source /opt/ros/humble/setup.bash \
    && cd ~/ \
    && curl -OL https://raw.githubusercontent.com/nandu-k01/robotic_arm_environment/main/docker_setup.sh
RUN chmod +x docker_setup.sh && ./docker_setup.sh
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
RUN echo "All Done "


# # Start with the ROS 2 base image
# FROM osrf/ros:humble-desktop

# # Install dependencies for CUDA
# RUN apt-get update && \
#     apt-get install -y \
#         gnupg2 \
#         lsb-release \
#         wget \
#         curl

# # Add NVIDIA package repositories
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
#     mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin | apt-key add - && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" | tee /etc/apt/sources.list.d/cuda.list && \
#     apt-get update

# # Install CUDA
# RUN apt-get install -y cuda

# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES \
#     ${NVIDIA_VISIBLE_DEVICES:-all}
# ENV NVIDIA_DRIVER_CAPABILITIES \
#     ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# # Set the shell to bash
# SHELL ["/bin/bash", "-c"]

# # Install Python dependencies
# RUN apt-get install -y \
#     python3-pip \
#     python3-colcon-common-extensions \
#     git

# # Setup ROS 2 workspace and install packages
# RUN source /opt/ros/humble/setup.bash \
#     && cd ~/ \
#     && curl -OL https://raw.githubusercontent.com/nandu-k01/robotic_arm_environment/main/docker_setup.sh \
#     && chmod +x docker_setup.sh \
#     && ./docker_setup.sh

# # Install PyTorch with CUDA support
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# # Setup environment variables for ROS 2
# RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
#     && echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

# # Confirm completion
# RUN echo "All Done"
