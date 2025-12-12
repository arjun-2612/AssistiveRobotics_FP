# AssistiveRobotics_FP
EECE 5552 Final Project for Team 4

Jorge Ortega, Arjun Viswanathan, Kyle Lou, Chengpeng Jin, Sasha Oswald.

![Force Control Demo](videos/force_control_grasp.mp4)

![RL Control Demo](videos/allegro_inhand.mp4)

# Cloning
Since this repository has submodules, make sure you clone recursively. In VS Code, when you do ```ctrl+shift+p``` and type ```clone``` it will show an option ```Clone (Recursive)```. Make sure you select that and paste the link to this repo.

# Conda Environment Setup

## MuJoCo

You will need to have a conda environment for this project. Install mujoco within the environment so it can run force control. You will also need Python=3.10. 

## IsaacLab

Only follow this section if you are on a local workstation.

First, you will need to make a conda environment wherever you want to install IsaacSim/Lab and the extensions in this repo. Consult this [```link```](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) on setting up IsaacSim and IsaacLab from the [```official repository```](https://github.com/isaac-sim/IsaacLab). Make sure you select the correct IsaacLab version (v2.0.0, not main). 

Once you have followed the link and installed IsaacSim/Lab, you need to install the extensions in this repo. Use the [```install_extensions.sh```](https://github.com/arjun-2612/AssistiveRobotics_FP/blob/main/isaac/install_extensions.sh) shell script as below from the main directory of the repo:

```
$ ./install_extensions.sh
```

Then, you can start training locally!

# Running the Code

NOTE: To generate logs of the run and visualize data generated, kill the code using ```CTRL+C``` in the terminal you run the commands from. For IsaacLab, set the number of environments to 1 for playing. 

## Force Control

To run force control in MuJoCo, following the commands below in your terminal. 

```
$ cd mujoco/allegro_hand
$ python ./test_grasp_hold.py
```

This will start the code and prompt you to hit ```ENTER``` on the keyboard to start MuJoCo. 

## RL Control

To train the RL policy, modify the [```yaml```](https://github.com/arjun-2612/AssistiveRobotics_FP/blob/main/isaac/source/isaac_extension/simulation/config/allegro.yaml) to select the task as ```Allegro-Cube-Repose-Train```, and number of environments you want. Set the mode to be ```train```. Then run this command

```
$ ./isaac_main.sh allegro
```

IsaacLab should start up and start training your policy. It will generate a log for you to visualize videos during the training process. 

To play the trained policy and visualize what it looks like after training, you will again modify the YAML file and select the task as ```Allegro-Cube-Repose-Play```, and set the mode as ```play```. Then run the same command as above. When IsaacSim starts up and shows you the results, it will generate an ```exported``` folder with the final policy file that can be readily used. 
