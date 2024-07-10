# Use for gripper URDF files generation: 
# different num of DOFs (different num of joints and links in each finger)
# different number of fingers
# different scale of fingers

import time
import os
import math
import random
from tqdm import tqdm
import trimesh


def scale_stl(stl_file, length_scale, width_scale, height_scale, output_file_name):
    stl_modified = trimesh.load(stl_file)
       
    scale_factors = (length_scale,width_scale,height_scale)
    stl_modified.apply_scale(scale_factors)

    output_dir = '/workspace/IsaacGymEnvs/assets/urdf/soft_gripper/mesh/modified_mesh/'
    output_path = os.path.join(output_dir, output_file_name)
    stl_modified.export(output_path)
    
stl_base_file = '/workspace/gripper_file_base/base.stl'
stl_mid_file = '/workspace/gripper_file_base/mid.stl'
stl_tip_file = '/workspace/gripper_file_base/tip.stl'


def scale_obj(obj_file,length_scale, width_scale, height_scale, output_file_name):
    
    obj_modified = trimesh.load(obj_file)
    scale_factors = (length_scale,width_scale,height_scale)
    obj_modified.vertices *= scale_factors
    
    output_dir = '/workspace/IsaacGymEnvs/assets/urdf/soft_gripper/mesh/modified_mesh/'
    output_path = os.path.join(output_dir, output_file_name)
    obj_modified.export(output_path)
         
    
obj_base_file = '/workspace/gripper_file_base/base.obj'
obj_mid_file = '/workspace/gripper_file_base/mid.obj'
obj_tip_file = '/workspace/gripper_file_base/tip.obj'



def generate_urdf_file(num_fingers, num_mids_per_finger, output_file_path, base_modified_obj_file, mid_modified_obj_file, tip_modified_obj_file,
                       base_modified_stl_file, mid_modified_stl_file, tip_modified_stl_file, length_scale):
    
    # modify the joint positions for scaled fingers
    base_joint_x = 0.0155*length_scale
    mid_joint_x = 0.0104*length_scale
    tip_joint_x = 0.0159*length_scale

    # Generate link and joint use to move the gripper in x,y,z direction
    urdf_template = f'''<?xml version="1.0" encoding="utf-8"?>
<robot name="gripper" xmlns:xacro="http://wiki.ros.org/xacro">

    <!-- base -->
    <link name="base"/>
    <link name="translation_x" />
    <link name="translation_y" />
    <link name="translation_z" />
    <link name="rotate_x" />
    <link name="rotate_y" />
    <link name="rotate_z" />
    <link name="gripper"/>

    <joint name="prismatic_joint_x" type="prismatic">
        <axis xyz="1 0 0"/>
        <origin xyz="0 0 0"/>
        <parent link="base"/>
        <child link="translation_x"/>
        <limit effort="50.0" lower="-2" upper="2" velocity="100"/>
    </joint>
    <joint name="prismatic_joint_y" type="prismatic">
        <axis xyz="0 1 0"/>
        <origin xyz="0 0 0"/>
        <parent link="translation_x"/>
        <child link="translation_y"/>
        <limit effort="50.0" lower="-2" upper="2" velocity="100"/>
    </joint>
    <joint name="prismatic_joint_z" type="prismatic">
        <axis xyz="0 0 1"/>
        <origin xyz="0 0 0"/>
        <parent link="translation_y"/>
        <child link="translation_z"/>
        <limit effort="50.0" lower="-2" upper="2" velocity="100"/>
    </joint>
    
    <joint name="rev_joint_x" type="revolute">
        <parent link="translation_z"/>
        <child link="rotate_x"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.57" upper="1.57" velocity="0.1" effort="12.0"/>
        <dynamics damping="10.0" friction="0.0"/>
    </joint>
    <joint name="rev_joint_y" type="revolute">
        <parent link="rotate_x"/>
        <child link="rotate_y"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" velocity="0.1" effort="12.0"/>
        <dynamics damping="10.0" friction="0.0"/>
    </joint>
    <joint name="rev_joint_z" type="revolute">
        <parent link="rotate_y"/>
        <child link="rotate_z"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" velocity="0.1" effort="12.0"/>
        <dynamics damping="10.0" friction="0.0"/>
    </joint>

    <joint name="fixed_joint" type="fixed">
        <parent link="rotate_z"/>
        <child link="gripper"/>
    </joint>
    
    <link name="gripper"/>
'''

    orient_dif = (2*math.pi/num_fingers)
    if length_scale > 2:
        radius = 0.06
    else:
        radius = 0.04

    # Generate base joints for each finger, (attach each finger into the "gripper" link)
    for finger_idx in range(num_fingers):
        angle_of_finger = finger_idx*orient_dif
        ## Randomize the angles between gripper's fingers
        # angle_of_finger = finger_idx * random.uniform(0, math.pi)
        # if num_fingers > 2:
        #     angle_of_finger += random.uniform(-math.pi/7, math.pi/7)
        urdf_template += f'''
    <joint name="{finger_idx}joint_base" type="fixed">
        <parent link="gripper"/>
        <child link="{finger_idx}link_0"/>
        
        <origin rpy="0 1.570796 {angle_of_finger}" xyz="{radius*math.cos(angle_of_finger)} {radius*math.sin(angle_of_finger)} 0"/>

    </joint>
'''
# origin for the gripper urdf to rotate to vertical pose instead of using gymapi.Transform() to rotate in isaac gym set_up
# Use the line below if you want to set the gripper in horizontal pose
# <origin rpy="{-angle_of_finger} 0 0" xyz="0 {radius*math.sin(angle_of_finger)} {radius*math.cos(angle_of_finger)}"/>
    
    # Generate finger's first link
    for finger_idx in range(num_fingers):
        urdf_template += f'''
    <!-- finger {finger_idx} -->
    <link name="{finger_idx}link_0">
        <visual>
            <geometry>
                <mesh filename="mesh/modified_mesh/{base_modified_obj_file}"/>
            </geometry>
            <material name="LightGrey">
                <color rgba="0.3 0.3 0.3 1.0"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <mesh filename="mesh/modified_mesh/{base_modified_stl_file}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
    </link>
    
    <joint name="{finger_idx}joint_0" type="revolute">
        <parent link="{finger_idx}link_0"/>
        <child link="{finger_idx}link_1"/>
        <origin rpy="0 0 0" xyz="{base_joint_x} 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-0.25" upper="0.25" velocity="0.1" effort="20.0"/>
        <dynamics damping="0.0" friction="0.0"/>
    </joint>
'''

        # Generate mid links and joints
        for mid_idx in range(1, num_mids_per_finger + 1):
            if mid_idx == num_mids_per_finger:
                x = tip_joint_x
            else:
                x = mid_joint_x
            urdf_template += f'''
    <link name="{finger_idx}link_{mid_idx}">
        <visual>
            <geometry>
                <mesh filename="mesh/modified_mesh/{mid_modified_obj_file}"/>
            </geometry>
            <material name="LightGrey">
                <color rgba="0.3 0.3 0.3 1.0"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <mesh filename="mesh/modified_mesh/{mid_modified_stl_file}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
    </link>

    <joint name="{finger_idx}joint_{mid_idx}" type="revolute">
        <parent link="{finger_idx}link_{mid_idx}"/>
        <child link="{finger_idx}link_{mid_idx+1}"/>
        <origin xyz="{x} 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-0.25" upper="0.25" velocity="0.1" effort="20.0"/>
        <dynamics damping="10.0" friction="0.0"/>
    </joint>
'''

        # Generate tip link and joint
        urdf_template += f'''
    <link name="{finger_idx}link_{num_mids_per_finger + 1}">
        <visual>
            <geometry>
                <mesh filename="mesh/modified_mesh/{tip_modified_obj_file}"/>
            </geometry>
            <material name="LightGrey">
                <color rgba="0.7 0.7 0.7 1.0"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <mesh filename="mesh/modified_mesh/{tip_modified_stl_file}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
    </link>
'''

    urdf_template += '</robot>'

    with open(output_file_path, 'w') as urdf_file:
        urdf_file.write(urdf_template)



def generate_urdf(number_fingers_list, number_mids_list, num_cases_per_setting):
    for num_fingers in number_fingers_list:
        for num_mids_per_finger in number_mids_list:
            # Create directories for each combination of fingers and mids
            output_folder = f"/workspace/IsaacGymEnvs/assets/urdf/soft_gripper/gen_grippers/{num_fingers}f_{num_mids_per_finger}m"
            os.makedirs(output_folder, exist_ok=True)
            
            for case in tqdm(range(num_cases_per_setting)):
                length_scale = random.uniform(2, 5)   # Scale x-axis
                width_scale = random.uniform(0.9, 2.5)    # Scale y-axis
                height_scale = random.uniform(0.9, 2.5)   # Scale z-axis

                # Generate different scale of finger's bases, mids, and tips
                
                base_modified_stl_file = f'base_modified_{num_fingers}f_{num_mids_per_finger}m_{case}.stl'
                mid_modified_stl_file = f'mid_modified_{num_fingers}f_{num_mids_per_finger}m_{case}.stl'
                tip_modified_stl_file = f'tip_modified_{num_fingers}f_{num_mids_per_finger}m_{case}.stl'
                
                scale_stl(stl_base_file, length_scale, width_scale, height_scale, base_modified_stl_file)
                scale_stl(stl_mid_file, length_scale, width_scale, height_scale, mid_modified_stl_file)
                scale_stl(stl_tip_file, length_scale, width_scale, height_scale, tip_modified_stl_file)
                
                base_modified_obj_file = f'base_modified_{num_fingers}f_{num_mids_per_finger}m_{case}.obj'
                mid_modified_obj_file = f'mid_modified_{num_fingers}f_{num_mids_per_finger}m_{case}.obj'
                tip_modified_obj_file = f'tip_modified_{num_fingers}f_{num_mids_per_finger}m_{case}.obj'
                
                scale_obj(obj_base_file, length_scale, width_scale, height_scale, base_modified_obj_file)
                scale_obj(obj_mid_file, length_scale, width_scale, height_scale, mid_modified_obj_file)
                scale_obj(obj_tip_file, length_scale, width_scale, height_scale, tip_modified_obj_file)
                
                output_file_path = os.path.join(output_folder, f"gripper_{case}.urdf")
                generate_urdf_file(num_fingers, num_mids_per_finger, output_file_path, base_modified_obj_file, mid_modified_obj_file, tip_modified_obj_file,
                       base_modified_stl_file, mid_modified_stl_file, tip_modified_stl_file, length_scale)
                # print(f"URDF file of {num_fingers}-fingered gripper with {num_mids_per_finger} DOFs generated successfully as gripper_{case}.")

number_fingers_list = list(range(2, 6))   # 2 to 5 fingers
number_mids_list = list(range(3, 6))       # 3 to 5 mids
num_cases_per_setting = 30

start = time.time()
generate_urdf(number_fingers_list, number_mids_list, num_cases_per_setting)
end = time.time()
print("Generating robots took %.3f seconds"
"" % (end - start))

