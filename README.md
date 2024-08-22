# Robotic Arm Environment for ROS 2 Humble

This repository, originally forked from [dvalenciar/robotic_arm_environment](https://github.com/dvalenciar/robotic_arm_environment), has been converted to work with ROS 2 Humble. The setup and usage have been streamlined for ease of use.

## Setup Instructions

To set up the environment, follow these steps:

1. **Download the Setup Script**

   Download the setup script using the following command:

   ```bash
   curl -OL https://raw.githubusercontent.com/nandu-k01/robotic_arm_environment/main/robotic_arm_setup.sh
   ```

2. **Run the Setup Script**

   Make the script executable and run it to install all the necessary dependencies and set up the environment:

   ```bash
   chmod +x robotic_arm_setup.sh
   ./robotic_arm_setup.sh
   ```

## Features

- **ROS 2 Humble Support**: The environment is now fully compatible with ROS 2 Humble.
- **Simplified Setup**: The setup process is streamlined with an easy-to-use script.
- **Enhanced Documentation**: Clear instructions are provided to get you started quickly.

## Getting Started

Once the setup is complete, you can start using the robotic arm environment in your ROS 2 Humble workspace. Ensure that your ROS environment is properly sourced:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

You can then launch the environment and start working on your robotic arm projects.

---

This guide should help you get started with the robotic arm environment in ROS 2 Humble. If you encounter any issues or need further assistance, please refer to the repository's documentation or reach out to the community.

