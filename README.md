# Soft Gripper Grasping Objects

This is the codebase for training soft grippers to grasp different objects in simulation.     
         
In this project, we want to train a policy that can generalize the grasping task to different types of soft-grippers and different object shapes. The task is currently being trained using only the gripper, without attachment to the robot arm. In the future, we aim to develop this further to integrate it with the real Franka robot arm.        

<p align="center">
  <img src="https://github.com/VKTS0912/soft-gripper-grasping-objects/assets/88523677/cb9ad3bd-6e7b-4c31-9dbd-6e95cadb8103">
</p>

## Description

Soft grippers are an emerging technology in robotic manipulation, offering a gentle and adaptable approach to grasping objects of various shapes, sizes, and fragilities. Unlike rigid grippers, soft grippers utilize flexible materials and compliant mechanisms, allowing them to conform to the surface of objects and apply uniform pressure without causing damage. This capability makes them ideal for applications in delicate tasks such as handling food, medical supplies, or other sensitive items.
<img align="right" src="https://github.com/VKTS0912/soft-gripper-grasping-objects/assets/88523677/239c08c7-c0df-4cb1-b4b0-ac135851a63e)">

However, training soft grippers to perform precise and reliable grasping tasks presents significant challenges. Traditional control methods often struggle to accommodate the complex dynamics and high degrees of freedom inherent in soft robotic systems. To address these challenges, the teacher-student paradigm has emerged as a powerful framework for training soft grippers effectively.     

Such a two-stage teacher-student training paradigm, where first a control policy is trained via Reinforcement Learning with full state information (teacher) and then a second student policy trained via supervised learning to mimic the teacher has been successfully used for many applications. For the grasping task of this project, the idea is to train the teacher policy using deep RL with privileged state information that can be extracted in simulation, and after that, a student policy can be trained to imitate the teacher using sensory inputs from the camera.

However, a significant challenge in robotic learning is the sim-to-real gap, the discrepancy between simulated environments and real-world conditions. While simulations provide a controlled and cost-effective platform for training robotic policies, they often fail to capture all the nuances of the physical world. Differences in sensor noise, friction, material properties, and dynamic interactions can lead to performance degradation when transferring a policy trained in simulation to a real-world robot.

So far, this project focuses on the initial stage of this process: training the teacher policy in simulation. By utilizing advanced simulation tools, this project aims to develop a robust and effective teacher policy that can demonstrate optimal grasping strategies which can generalize to different scale of soft grippers and objects.

## Installation

### Dependencies

* [Issac Gym](https://developer.nvidia.com/isaac-gym)
* [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
* [Wandb](https://wandb.ai/site)
* [Trimesh](https://github.com/mikedh/trimesh)

### Download packages

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. It is highly recommended to use a conda environment to simplify set up.     

Ensure that Isaac Gym works on your system by running one of the examples from the ``python/examples`` directory, like ``joint_monkey.py``. Follow troubleshooting steps described in the Isaac Gym Preview 4 install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install the IsaacGymEnvs repo:
```
cd IsaacGymEnvs
pip install -e .
```
The directory should look like this:    
```
-- Root
---- IsaacGymEnvs
---- isaacgym
```
Please review the Isaac Gym Benchmark Environments from the [website](https://github.com/isaac-sim/IsaacGymEnvs) for details on setting up the environment and training example tasks.   
### Download the repository
Please clone the repo to your root/home folder.       
```     
git clone https://github.com/VKTS0912/soft-gripper-grasping-objects.git
```
### Download the object dataset
Create a new folder named ``egad_objects`` inside the following folder: ``/workspace/IsaacGymEnvs/assets/urdf``.          
```
cd /workspace/IsaacGymEnvs/assets/urdf
mkdir egad_objects
```
Download the object dataset from [here](https://dougsm.github.io/egad/), and unzip it to the created folder: ``/workspace/IsaacGymEnvs/assets/urdf/egad_objects``.    
The object .obj files must be found in a folder directory like this:
```
-- Root
---- egad_objects
----- egadtrainset
------ egad_train_set
```
Modify the following line 9 and 44 in file ``gen_objects.py`` to make the correct folder workspace (replace ``workspace`` with the correct root directory on your machine):             
``9. object_folder = '/workspace/IsaacGymEnvs/assets/urdf/egad_objects/egadtrainset/egad_train_set'``               
``44. output_folder = f"/workspace/IsaacGymEnvs/assets/urdf/egad_objects"``          
Run the file and you will see the created urdf files of the object assets located in ``/workspace/IsaacGymEnvs/assets/urdf/egad_objects``.             
## Set up the task
### Prepare gripper assets
We use the base urdf mesh files of 3-fingered gripper to generate different types of grippers.     
First, create output folders for the generated grippers:   
```
cd /IsaacGymEnvs/assets/urdf
mkdir -p soft_gripper/mesh/modified_mesh
```
Locate the folder named ``gripper_file_base`` that is downloaded from the repository, make sure the directory looks like this:
```
-- Root
---- IsaacGymEnvs
---- gripper_file_base
```
Modify the file ``gen_grippers.py`` to make the correct folder workspace (replace ``workspace`` with the correct root directory on your machine).          
Then run the file and you will see the created urdf files of grippers located in ``/workspace/IsaacGymEnvs/assets/urdf/soft_gripper/gen_grippers``.         
** If you want to use the pre-generated gripper assets, unzip the file ``soft-gripper.zip`` which can be downloaded from this repo instead of creating the folder as above.          
### Prepare the training environment  
#### Tasks
Source code for tasks can be found in ``IsaacGymEnvs/isaacgymenvs/tasks``. Move the task file ``soft_gripper.py`` downloaded from the repo to this folder.       
In this folder, open ``__init__.py`` and add these lines to initiate the task:
```
from .soft_gripper import SoftGripper
isaacgym_task_map = {
...
"SoftGripper": SoftGripper,
}
```
#### Config
Every task requires config files for training. Move the downloaded task-config file ``SoftGripper.yaml`` to ``/IsaacGymEnvs/isaacgymenvs/cfg/task``, and the train-config file ``SoftGripperPPO.yaml`` to ``/IsaacGymEnvs/isaacgymenvs/cfg/train``.        
Here we use the default Isaac Gym Benchmark config for the PPO algorithm used for training the teacher policy.            
### Executing program
* How to run the program        
To train the grasping policy, run this line:
```
cd IsaacGymEnvs/isaacgymenvs
python train.py task=SoftGripper
```
The training should be running now. By default, the task is training multiple different scale of soft grippers to grasp a cube. You can modify the amount of grippers and objects in the task file to test the policy.     
Hit the ``v`` key while running to disable viewer updates and allow training to proceed faster. Hit ``v`` again to resume viewing. Or you can train headlessly:
```
python train.py task=SoftGripper headless=True
```
* Plot the results           
We use ``wandb`` to plot additional results while training for viewing and debugging. Modify the task file by uncommenting the lines in function ``def compute_reward(self, actions)``.                  
Each task subclasses the VecEnv base class in ``isaacgymenvs/tasks/base/vec_task.py``. Open this file and in ``def allocate_buffers(self)``, add these lines:
```
self.env_steps = 0
self.successes = torch.zeros(self.num_envs, dtype=torch.float,device=self.device)
self.success_rate = torch.zeros(1, dtype=torch.float,device=self.device)
```
Make sure ``test`` is set to ``False`` in the task config file ``SoftGripper.yaml`` to plot the results while training. Then you can run this to sync your results to wandb:
```
python train.py task=SoftGripper wandb_activate=True wandb_entity=[your_wandb_account_name] wandb_project=[your_project_name]
```
### Pre-trained models
We provide the pre-trained model for the teacher in the repo. The model was trained using the gripper assets in this repo.
* Usage      
You can test the model with the pre-generated gripper asset unzipped from ``soft-gripper.zip`` which we use for training. Or you can load different gripper assets and see the policy's performance.        
Whenever you train a task, checkpoints are saved in the folder ``runs/EXPERIMENT_NAME/nn``. If you haven't trained any task and folder ``runs`` is not created yet, create it first by:
```
cd IsaacGymEnv/isaacgymenvs
mkdir runs
```
Move the folder ``Pretrained`` to ``/workspace/IsaacGymEnvs/isaacgymenvs/runs``.    
Load the trained checkpoint and see the pre-trained model:
```
cd IsaacGymEnvs/isaacgymenvs
python train.py task=SoftGripper checkpoint=runs/Pretrained/soft_gripper_pretrained/nn/last_SoftGripper_ep_10000_rew__2084.56_.pth test=True num_envs=50
```
## Authors

Contributors names and contact info            

[Son Vo] [email](mailto:son.vkt202768@sis.hust.edu.vn)      
[Khuyen Gia Pham] [email](mailto:21khuyen.pg@vinuni.edu.vn)

## Acknowledgments
I would like to express my sincere gratitude to my supervisor, [Acar Cihan](https://scholar.google.co.nz/citations?user=4oQMp4EAAAAJ&hl=en), for his invaluable guidance and support throughout this project. His insights and expertise have been instrumental in the successful completion of our work.

Please review these documents for reference.
* [Visual Dexterity](https://arxiv.org/abs/2211.11744)
* [Hardware Conditioned Policy](https://arxiv.org/abs/1811.09864)

