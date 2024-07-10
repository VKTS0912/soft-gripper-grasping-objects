# Use for object files generation: 

import time
import os
import math
import random
from tqdm import tqdm

object_folder = '/workspace/IsaacGymEnvs/assets/urdf/egad_objects/egadtrainset/egad_train_set'
obj_files =[]
for filename in os.listdir(object_folder):
    file_path = os.path.join("egadtrainset/egad_train_set/", filename)
    obj_files.append(file_path)
    
# for i in range(len(obj_files)):
#     print(obj_files[i])
def generate_object(obj_files, output_file_path, i, scale, mass):
    urdf_template = f'''<?xml version="1.0"?>
    
<robot name="egad_object">
    <link name="object">
        <inertial>
            <mass value= "{mass}"/>
            <origin xyz="0 0 0" />
            <inertia ixx="0.0002" ixy="0.0" ixz="0.0" iyy="0.0002" iyz="0.0" izz="0.0002"/>
        </inertial>
        <visual>
            <geometry>
                <mesh filename= "{obj_files[i]}" scale="{scale} {scale} {scale}"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <mesh filename= "{obj_files[i]}" scale="{scale} {scale} {scale}"/>
            </geometry>
        </collision>
    </link>
</robot>
    '''
    
    with open(output_file_path, 'w') as urdf_file:
        urdf_file.write(urdf_template)

output_folder = f"/workspace/IsaacGymEnvs/assets/urdf/egad_objects"
os.makedirs(output_folder, exist_ok=True)

# Vary the scale to change the size of objects
scale = 0.0016  # 0.0011
start = time.time()

for i in tqdm(range(len(obj_files))):  
    mass = random.uniform(0.05,0.15) 
    output_file_path = os.path.join(output_folder, f"object_{i}.urdf")
    generate_object(obj_files, output_file_path, i, scale, mass)

end = time.time()
print("Generating objects took %.3f seconds"
"" % (end - start))

