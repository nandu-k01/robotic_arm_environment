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
