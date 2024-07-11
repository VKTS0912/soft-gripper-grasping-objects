import numpy as np
import os
import torch
import random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask
from tqdm import tqdm
import wandb

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
   """
   Converts scaled axis-angle to quat.
   Args:
       vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
       eps (float): Stability value below which small values will be mapped to 0


   Returns:
       tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
   """
   # type: (Tensor, float) -> Tensor
   # store input shape and reshape
   input_shape = vec.shape[:-1]
   vec = vec.reshape(-1, 3)

   # Grab angle
   angle = torch.norm(vec, dim=-1, keepdim=True)

   # Create return array
   quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
   quat[:, 3] = 1.0

   # Grab indexes where angle is not zero an convert the input to its quaternion form
   idx = angle.reshape(-1) > eps
   quat[idx, :] = torch.cat([
       vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
       torch.cos(angle[idx, :] / 2.0)
   ], dim=-1)

   # Reshape and return output
   quat = quat.reshape(list(input_shape) + [4, ])
   return quat

class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

class SoftGripper(VecTask):
   def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
       self.cfg = cfg

       self.max_episode_length = self.cfg["env"]["episodeLength"]


       self.action_scale = self.cfg["env"]["actionScale"]
       self.start_position_noise = self.cfg["env"]["startPositionNoise"]
       self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
       self.gripper_position_noise = self.cfg["env"]["gripperPositionNoise"]
       self.gripper_rotation_noise = self.cfg["env"]["gripperRotationNoise"]
       self.gripper_dof_noise = self.cfg["env"]["gripperDofNoise"]
       self.aggregate_mode = self.cfg["env"]["aggregateMode"]


       # Create dicts to pass to reward function
       self.reward_settings = {
           "r_dist_scale": self.cfg["env"]["distRewardScale"],
           "r_lift_scale": self.cfg["env"]["liftRewardScale"],
           "r_goal_scale": self.cfg["env"]["goalRewardScale"],
       }


       # obs include: object pos and orn(7), gripper pos and orn (7) + q_gripper (1) + object_id (1)
       self.cfg["env"]["numObservations"] = 16 
       # actions include: 6 for gripper's base (movement in xyz and rpy) and 1 for gripping (open or close the fingers)
       self.cfg["env"]["numActions"] = 7 
       # this code is considering the 3-fingered 4-mid soft gripper with 21 DOFs (6 dofs for movement and 15 joints)


       # Values to be filled in at runtime
       self.states = {}                        # will be dict filled with relevant states to use for reward calculation
       self.handles = {}                       # will be dict mapping names to relevant sim handles
       self.num_dofs = None                    # Total number of DOFs per env
       self.actions = None                     # Current actions to be deployed
       self._init_object_state = None           # Initial state of object for the current env
       self._object_state = None                # Current state of object for the current env
       self._object_id = None                   # Actor ID corresponding to object for a given env
       
        
       self._dof_state = None  # State of all joints       (n_envs, n_dof)
       # Tensor placeholders
       self._root_state = None             # State of root body        (n_envs, 13)
       self._q = None  # Joint positions           (n_envs, n_dof)
       self._qd = None                     # Joint velocities          (n_envs, n_dof)
       self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
       self._contact_forces = None     # Contact forces in sim
       self._eef_state = None  # end effector state (at grasping point)
       self._eef_f1_state = None  # end effector state (at 1st fingertip)
       self._eef_f2_state = None  # end effector state (at 2nd fingertip)
       self._eef_f3_state = None  # end effector state (at 3rd fingertip)

       self._base_control = None  # Tensor buffer for controlling base
       self._finger_control = None  # Tensor buffer for controlling fingers
       self._pos_control = None            # Position actions
       self._effort_control = None         # Torque actions
       self._gripper_effort_limits = None        # Actuator effort limits for gripper
       self._global_indices = None         # Unique indices corresponding to all envs in flattened array
       self.debug_viz = self.cfg["env"]["enableDebugVis"]


       self.up_axis = "z"
       self.up_axis_idx = 2
       
       self.test_mode = self.cfg["test"]

       super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)


       # (21 DOFs = 15 joint + 6DOFs of moving)
       self.gripper_default_dof_pos = to_torch(
           [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], device=self.device)
      
       self.xy_center = to_torch([0, 0], dtype=torch.float, device=self.device)     # object goal to stay near the center
       self.goal = to_torch([0, 0, 1.25], dtype=torch.float, device=self.device)    # Goal to lift the object
       self.quat = to_torch([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=self.device)   # Vertical orientation after successful grasp
       
       # Set control limits (for gripper movement)
       self.cmd_limit = to_torch([0.1]*6, device=self.device).unsqueeze(0)
       
       # Reset all environments
       self.reset_idx(torch.arange(self.num_envs, device=self.device))

       # Refresh tensors
       self._refresh()


   def create_sim(self):
       self.sim_params.up_axis = gymapi.UP_AXIS_Z
       self.sim_params.gravity.x = 0
       self.sim_params.gravity.y = 0
       self.sim_params.gravity.z = -9.81
       self.sim = super().create_sim(
           self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
       self._create_ground_plane()
       self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


   def _create_ground_plane(self):
       plane_params = gymapi.PlaneParams()
       plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
       self.gym.add_ground(self.sim, plane_params)


   def create_robot_asset(self, asset_root):
        # Define the folder containing the URDF files
        robot_urdf_folder = '/home/samuel/IsaacGymEnvs/assets/urdf/soft_gripper/gen_grippers/3f_4m'

        # Initialize an empty list to store AssetDesc objects
        robot_asset_descriptors = []
        
        for robot_filename in os.listdir(robot_urdf_folder):
            if robot_filename.endswith('.urdf'):  # Filter only URDF files
                robot_file_path = os.path.join("urdf/soft_gripper/gen_grippers/3f_4m", robot_filename)
                
                # Create an AssetDesc object and append it to the list
                robot_asset_desc = AssetDesc(robot_file_path, False)
                robot_asset_descriptors.append(robot_asset_desc)
                      
        # Loading Robot Assets
        robot_assets = []
        print("Loading robot asset from '%s'" % (asset_root))
        for robot_asset_desc in tqdm(robot_asset_descriptors):
            robot_asset_file = robot_asset_desc.file_name
            # print("robot asset_file: ", robot_asset_file)
            robot_asset_options = gymapi.AssetOptions()
            robot_asset_options.fix_base_link = True
            robot_asset_options.flip_visual_attachments = robot_asset_desc.flip_visual_attachments
            robot_asset_options.disable_gravity = True
            robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
            robot_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
            robot_asset_options.use_mesh_materials = True
            robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, robot_asset_options)
            # print("Loading asset '%s' from '%s'" % (robot_asset_file, asset_root))
            robot_assets.append(robot_asset)
            
        self.num_robot_assets = len(robot_assets)
        return robot_assets
        
   def create_object_asset(self, asset_root):
        # Define the folder containing the URDF files
        object_urdf_folder = '/home/samuel/IsaacGymEnvs/assets/urdf/egad_objects'
        
        # Initialize an empty list to store AssetDesc objects
        object_asset_descriptors = []  
        
        for object_filename in os.listdir(object_urdf_folder):
            if object_filename.endswith('.urdf'):  # Filter only URDF files
                object_file_path = os.path.join("urdf/egad_objects", object_filename)
                # Create an AssetDesc object and append it to the list
                object_asset_desc = AssetDesc(object_file_path, False)
                object_asset_descriptors.append(object_asset_desc)
                
        # Loading Object Assets 
        object_assets = []
        print("Loading object asset from '%s'" % (asset_root))
        for object_asset_desc in tqdm(object_asset_descriptors):
            object_asset_file = object_asset_desc.file_name
            # print("object asset_file: ", object_asset_file)
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.fix_base_link = False
            # Enable convex decomposition
            object_asset_options.vhacd_enabled = False
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 100000
            object_asset_options.flip_visual_attachments = object_asset_desc.flip_visual_attachments
            object_asset_options.use_mesh_materials = True
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            # print("Loading asset '%s' from '%s'" % (object_asset_file, asset_root))
            object_assets.append(object_asset)
            
        self.num_object_assets = len(object_assets)
        return object_assets
    
   def _create_envs(self, num_envs, spacing, num_per_row):
       lower = gymapi.Vec3(-spacing, -spacing, 0.0)
       upper = gymapi.Vec3(spacing, spacing, spacing)


       asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
       gripper_asset_file = "urdf/soft_gripper/gen_grippers/3f_4m/gripper_0.urdf"
       
       if "asset" in self.cfg["env"]:
           asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
           gripper_asset_file = self.cfg["env"]["asset"].get("assetFileNameSoftGripper", gripper_asset_file)

       ## Choose to train with 1 gripper or many different grippers
       # Load multiple grippers with different scale
       robot_assets = self.create_robot_asset(asset_root)
        
       # Load only 1 type of gripper asset for all environments
       gripper_asset_options = gymapi.AssetOptions()
       # gripper_asset_options.flip_visual_attachments = False
       gripper_asset_options.fix_base_link = True
       gripper_asset_options.use_mesh_materials = True
       gripper_asset_options.disable_gravity = True
       gripper_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
       gripper_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
    #    gripper_asset = self.gym.load_asset(self.sim, asset_root, gripper_asset_file, gripper_asset_options)
       
       # Joints of gripper has their own stiffness and damping
       gripper_dof_stiffness = (8/5)*to_torch([5000.,]*21, dtype=torch.float, device=self.device)
       gripper_dof_damping = to_torch([5.0e2,]*21, dtype=torch.float, device=self.device)
       # Create table asset
       table_pos = [0.0, 0.0, 1.0]
       table_thickness = 0.05
       table_opts = gymapi.AssetOptions()
       table_opts.fix_base_link = True
       table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)


       # Create object asset
       object_opts = gymapi.AssetOptions()
       object_opts.fix_base_link = False
    #    object_opts.disable_gravity = False
    #    object_opts.use_mesh_materials = True
       
    ## Choose to train with only 1 object or different objects
    # Load multiple objects from egad object dataset
    #    object_assets = self.create_object_asset(asset_root)
       
    # Load 1 object from egad object dataset
    #    object_file = "urdf/egad_big_objects/object_0.urdf"
    #    object_asset = self.gym.load_asset(self.sim, asset_root, object_file, object_opts)
    
    ## Or using simple default objects in isaacgym
    # Create a sphere
    #    radius = 0.08
    #    object_asset = self.gym.create_sphere(self.sim, radius, object_opts)
    # Create a box  
       box_size = 0.11 #0.085
       object_asset = self.gym.create_box(self.sim, *([box_size] * 3), object_opts)
       
       object_color = gymapi.Vec3(0.9, 0.8, 0.3)


    ##    Uncomment to set gripper dof properties if using only 1 type of gripper for all environments
    #    self.num_gripper_bodies = self.gym.get_asset_rigid_body_count(gripper_asset)
    #    self.num_gripper_dofs = self.gym.get_asset_dof_count(gripper_asset)

    # #    print("num gripper bodies: ", self.num_gripper_bodies)
    # #    print("num gripper dofs: ", self.num_gripper_dofs)

    #    gripper_dof_props = self.gym.get_asset_dof_properties(gripper_asset)
    #    self.gripper_dof_lower_limits = []
    #    self.gripper_dof_upper_limits = []
    #    self._gripper_effort_limits = []
    #    for i in range(self.num_gripper_dofs):
    #        gripper_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i < 6 else gymapi.DOF_MODE_EFFORT
    #     #    gripper_dof_props['upper'][i] = 0.13
    #     #    gripper_dof_props['lower'][i] = -0.18
    #        if self.physics_engine == gymapi.SIM_PHYSX:
    #            gripper_dof_props['stiffness'][i] = gripper_dof_stiffness[i]
    #            gripper_dof_props['damping'][i] = gripper_dof_damping[i]
    #        else:
    #            gripper_dof_props['stiffness'][i] = 7000.0
    #            gripper_dof_props['damping'][i] = 500.0


    #        self.gripper_dof_lower_limits.append(gripper_dof_props['lower'][i])
    #        self.gripper_dof_upper_limits.append(gripper_dof_props['upper'][i])
    #        self._gripper_effort_limits.append(gripper_dof_props['effort'][i])
    #    self.gripper_dof_lower_limits = to_torch(self.gripper_dof_lower_limits, device=self.device)
    #    self.gripper_dof_upper_limits = to_torch(self.gripper_dof_upper_limits, device=self.device)
    #    self._gripper_effort_limits = to_torch(self._gripper_effort_limits, device=self.device)

    #    self.gripper_dof_lower_limits = to_torch(self.gripper_dof_lower_limits, device=self.device)
    #    self.gripper_dof_upper_limits = to_torch(self.gripper_dof_upper_limits, device=self.device)
    #    self._gripper_effort_limits = to_torch(self._gripper_effort_limits, device=self.device)
    #    gripper_dof_props['effort'][:6] = 400
    #    gripper_dof_props['effort'][6:] = 800


    #    Get properties of different types of grippers
       for gripper_asset in robot_assets:   
            self.num_gripper_dof = self.gym.get_asset_dof_count(gripper_asset)  
            self.num_gripper_bodies = self.gym.get_asset_rigid_body_count(gripper_asset)
            
            # print("num of gripper bodies: ", self.num_gripper_bodies)
            # print("num of gripper dofs: ", self.num_gripper_dof)
            
            gripper_dof_props = self.gym.get_asset_dof_properties(gripper_asset)
            self.gripper_dof_lower_limits = []
            self.gripper_dof_upper_limits = []
            self.gripper_effort_limits = []
            
            for i in range(self.num_gripper_dof):
                gripper_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i < 6 else gymapi.DOF_MODE_EFFORT
                if self.physics_engine == gymapi.SIM_PHYSX:
                    gripper_dof_props['stiffness'][i] = gripper_dof_stiffness[i]
                    gripper_dof_props['damping'][i] = gripper_dof_damping[i]
                else:
                    gripper_dof_props['stiffness'][i] = 7000.0
                    gripper_dof_props['damping'][i] = 500.0
                    
                self.gripper_dof_lower_limits.append(gripper_dof_props['lower'][i])
                self.gripper_dof_upper_limits.append(gripper_dof_props['upper'][i])
                self.gripper_effort_limits.append(gripper_dof_props["effort"][i])

            
            self.gripper_dof_lower_limits = to_torch(self.gripper_dof_lower_limits, device=self.device)
            self.gripper_dof_upper_limits = to_torch(self.gripper_dof_upper_limits, device=self.device)
            self.gripper_effort_limits = to_torch(self.gripper_effort_limits, device=self.device)
            
            gripper_dof_props['effort'][:6] = 400
            gripper_dof_props['effort'][6:] = 800
            
        
       # Define start pose for gripper
       gripper_start_pose = gymapi.Transform()
       gripper_start_pose.p = gymapi.Vec3(0.0 ,0, 0.5)
       gripper_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

       # Define start pose for table
       table_start_pose = gymapi.Transform()
       table_start_pose.p = gymapi.Vec3(*table_pos)
       table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
       self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
       self.reward_settings["table_height"] = self._table_surface_pos[2]


       # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
       object_start_pose = gymapi.Transform()
       object_start_pose.p = gymapi.Vec3(-1.0, 0.0, self._table_surface_pos[2] + 0.01)
       object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
       

       self.grippers = []
       self.envs = []
       self.object_index = []
    
       # Create environments
       for i in range(self.num_envs):
           # create env instance
           env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
           # Create actors 
           # NOTE: gripper should ALWAYS be loaded first in sim!

           # Create gripper
           # Potentially randomize start pose
           if self.gripper_position_noise > 0:
               rand_xy = self.gripper_position_noise * (-1. + np.random.rand(2) * 2.0)
               gripper_start_pose.p = gymapi.Vec3(0 + rand_xy[0], 0.0 + rand_xy[1],
                                                2)
           if self.gripper_rotation_noise > 0:
               rand_rot = torch.zeros(1, 3)
               rand_rot[:, -1] = self.gripper_rotation_noise * (-1. + np.random.rand() * 2.0)
               new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
               gripper_start_pose.r = gymapi.Quat(*new_quat)

           # Create gripper actor
           # Create actors for multiple gripper assets 
           gripper_asset_index = i % self.num_robot_assets
           gripper_asset = robot_assets[gripper_asset_index]
        #    gripper_asset = random.choice(robot_assets)
           gripper_actor = self.gym.create_actor(env_ptr, gripper_asset, gripper_start_pose, f"Gripper No. {i}", i, 0, 0)
           
        #    self.gym.set_actor_scale(env_ptr, gripper_actor, 1.5)
           self.gym.set_actor_dof_properties(env_ptr, gripper_actor, gripper_dof_props)
                            

           # Create table
           table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)


           # Create object actors for multiple object assets
        
        #    object_asset_index = i % self.num_object_assets
        #    object_asset = object_assets[object_asset_index]
        #    self.object_index.append(object_asset_index)
        
        # Create 1 object actor only
           self.object_index.append(0)  # object_index=0 because of training only 1 object
           self._object_id = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 2, 0)
           # Set colors
           self.gym.set_rigid_body_color(env_ptr, self._object_id, 0, gymapi.MESH_VISUAL, object_color)
        
           # Store the created env pointers
           self.envs.append(env_ptr)
           self.grippers.append(gripper_actor)
           
       # Setup init state buffer
       self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)
       
       # Reshape the object index to have shape (num_env, 1)
       self.object_index = to_torch(self.object_index, device=self.device).unsqueeze(1)

       # Setup data
       self.init_data()
       

   def init_data(self):
       self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
       # Setup sim handles
       env_ptr = self.envs[0]
       gripper_handle = 0
       self.handles = {
           # Gripper
           "gripper": self.gym.find_actor_rigid_body_handle(env_ptr, gripper_handle, 'gripper'),
           "finger1_tip": self.gym.find_actor_rigid_body_handle(env_ptr, gripper_handle, "0link_5"),
           "finger2_tip": self.gym.find_actor_rigid_body_handle(env_ptr, gripper_handle, "1link_5"),
           "finger3_tip": self.gym.find_actor_rigid_body_handle(env_ptr, gripper_handle, "2link_5"),
           
            # Object
           "object_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._object_id, "object"),
       }
       
       # Setup tensor buffers
       _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
       _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
       _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
       self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
       self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
       self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
       
       self._q = self._dof_state[..., 0]
       self._qd = self._dof_state[..., 1]
       self._eef_state = self._rigid_body_state[:, self.handles["gripper"], :]
       self._eef_f1_state = self._rigid_body_state[:, self.handles["finger1_tip"], :]
       self._eef_f2_state = self._rigid_body_state[:, self.handles["finger2_tip"], :]
       self._eef_f3_state = self._rigid_body_state[:, self.handles["finger3_tip"], :]
       self._object_state = self._root_state[:, self._object_id, :]
           
                
       # Initialize states
       # self.states.update({
       #     "object_size": torch.ones_like(self._eef_state[:, 0]) * self.object_size,
       # })


       # Initialize actions
       self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
       self._effort_control = torch.zeros_like(self._pos_control)


       # Initialize control
       self._base_control = self._pos_control[:, :6]
       self._finger_control = self._effort_control[:, 6:]
       # Initialize indices
       self._global_indices = torch.arange(self.num_envs*3, dtype=torch.int32,
                                          device=self.device).view(self.num_envs, -1)
         
    
   def _update_states(self):
       self.states.update({
           # Franka
           "q": self._q[:, :],
           "q_gripper": self._q[:, 6:7],
           "eef_pos": self._eef_state[:, :3],
           "eef_quat": self._eef_state[:, 3:7],
           "eef_vel": self._eef_state[:, 7:],
           "eef_f1_pos": self._eef_f1_state[:, :3],
           "eef_f2_pos": self._eef_f2_state[:, :3],
           "eef_f3_pos": self._eef_f3_state[:, :3],
                  
           # Objects
           "object_quat": self._object_state[:, 3:7],
           "object_pos": self._object_state[:, :3],
           "object_linvels": self._object_state[:, 7:10],
           "object_pos_relative": self._object_state[:, :3] - self._eef_state[:, :3],
           
           "object_index": self.object_index,
           
       })


   def _refresh(self):
       self.gym.refresh_actor_root_state_tensor(self.sim)
       self.gym.refresh_dof_state_tensor(self.sim)
       self.gym.refresh_rigid_body_state_tensor(self.sim) 

       self._update_states()


   def compute_reward(self, actions):
       # self._refresh()
       self.rew_buf[:], self.reset_buf[:], dist_reward, lift_reward, goal_reward, quat_reward, pos_object_reward, self.successes[:], self.success_rate[:] = compute_gripper_reward(
           self.reset_buf, self.progress_buf, self.successes, self.success_rate, self.xy_center, self.goal, self.quat, self.actions, self.states, self.reward_settings, self.max_episode_length)
       
    #    if (not self.test_mode):
    #        # log the success rate
    #        wandb.log({"number_of_success": torch.sum(self.successes).item()}, step=self.env_steps)
    #        wandb.log({"success_rate": self.success_rate.item()}, step=self.env_steps)
        
    #        # log individual reward for debugging
    #        wandb.log({"dist_reward": torch.mean(dist_reward)}, step=self.env_steps)
    #        wandb.log({"lift_reward": torch.mean(lift_reward)}, step=self.env_steps)
    #        wandb.log({"goal_reward": torch.mean(goal_reward)}, step=self.env_steps)
    #        wandb.log({"quat_reward": torch.mean(quat_reward)}, step=self.env_steps)
    #        wandb.log({"pos_object_reward": torch.mean(pos_object_reward)}, step=self.env_steps)
        
    #        wandb.log({"object_height": torch.mean(self.states["object_pos"][:, 2])})
    #        wandb.log({"object_dis": torch.mean(self.states["object_pos"][:, 2]-self.reward_settings["table_height"])}) 
        
    #        wandb.log({"d_goal": torch.mean(1.5 - self.states["object_pos"][:, 2])})
    #        wandb.log({"quat_difference": torch.norm(self.states["eef_quat"] - self.quat)})
    #    else:
    #        # if it is in test mode, then not plot the result
    #        pass
           
    

   def compute_observations(self):
       self._refresh()
       obs = ["object_index", "object_quat", "object_pos", "eef_pos", "eef_quat"]
       obs += ["q_gripper"]
       
       self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

       maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}
       return self.obs_buf


   def reset_idx(self, env_ids):
    #    self._refresh()
       env_ids_int32 = env_ids.to(dtype=torch.int32)

       self._reset_init_object_state(object='object', env_ids=env_ids, check_valid=False)

       # Write these new init states to the sim states
       self._object_state[env_ids] = self._init_object_state[env_ids]


       # Reset agent
       reset_noise = torch.rand((len(env_ids), 21), device=self.device)
       pos = tensor_clamp(
           self.gripper_default_dof_pos.unsqueeze(0) +
           self.gripper_dof_noise * 2.0 * (reset_noise - 0.5),
           self.gripper_dof_lower_limits.unsqueeze(0), self.gripper_dof_upper_limits.unsqueeze(0))


       # Overwrite gripper init pos (no noise since these are always position controlled)
       pos[:, 6:] = self.gripper_default_dof_pos[6:]


       # Reset the internal obs accordingly
       self._q[env_ids, :] = pos
       self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])


       # Set any position control to the current position, and any vel / effort control to be 0
       # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
       self._pos_control[env_ids, :] = pos
       self._effort_control[env_ids, :] = torch.zeros_like(pos)


       # Deploy updates
       multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
       self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                       gymtorch.unwrap_tensor(self._pos_control),
                                                       gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                       len(multi_env_ids_int32))
       self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                       gymtorch.unwrap_tensor(self._effort_control),
                                                       gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                       len(multi_env_ids_int32))
       self.gym.set_dof_state_tensor_indexed(self.sim,
                                             gymtorch.unwrap_tensor(self._dof_state),
                                             gymtorch.unwrap_tensor(multi_env_ids_int32),
                                             len(multi_env_ids_int32))


       # Update object states
       multi_env_ids_object_int32 = self._global_indices[env_ids, -1].flatten()
       self.gym.set_actor_root_state_tensor_indexed(
           self.sim, gymtorch.unwrap_tensor(self._root_state),
           gymtorch.unwrap_tensor(multi_env_ids_object_int32), len(multi_env_ids_object_int32))


       self.progress_buf[env_ids] = 0
       self.reset_buf[env_ids] = 0


   def _reset_init_object_state(self, object, env_ids, check_valid=True):
       # If env_ids is None, we reset all the envs
       if env_ids is None:
           env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

       num_resets = len(env_ids)
       sampled_object_state = torch.zeros(num_resets, 13, device=self.device)

       # Get correct references depending on which one was selected
       this_object_state_all = self._init_object_state
       # object_heights = self.states["object_size"]

       # Sampling is "centered" around middle of table
       centered_object_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)
       # Set z value, which is fixed height
       sampled_object_state[:, 2] = self._table_surface_pos[2] + 0.01 #object_heights.squeeze(-1)[env_ids] / 2

       # Initialize rotation, which is no rotation (quat w = 1)
       sampled_object_state[:, 6] = 1.0


       # We just directly sample XY positions with noise
       sampled_object_state[:, :2] = centered_object_xy_state.unsqueeze(0) + \
                                               2.0 * self.start_position_noise * (
                                                       torch.rand(num_resets, 2, device=self.device) - 0.5)

       # Sample rotation value
       if self.start_rotation_noise > 0:
           aa_rot = torch.zeros(num_resets, 3, device=self.device)
           aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
           sampled_object_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_object_state[:, 3:7])

       # Lastly, set these sampled values as the new init state
       this_object_state_all[env_ids, :] = sampled_object_state


   def pre_physics_step(self, actions):

       self.actions = actions.clone().to(self.device)
       # Split base and gripper command
       u_base, u_gripper = self.actions[:, :-1], self.actions[:, -1]

       # Control base (scale value first)
       u_base = u_base / self.action_scale
       u_base = tensor_clamp(u_base.squeeze(-1),
                        -self.cmd_limit, self.cmd_limit)

       q_new = self._q[:, :6] + u_base
       self._base_control[:, :] = q_new

       # Control gripper
       u_fingers = torch.zeros_like(self._finger_control)

       # Calculate the tensor for the true branch
       true_branch_tensor = torch.full_like(self._finger_control, 400)

       # Calculate the tensor for the false branch
       false_branch_tensor = torch.full_like(self._finger_control, -400)

       u_gripper = u_gripper.unsqueeze(1) 

       u_gripper_expand = abs(u_gripper.expand(-1, self._finger_control.size(1))) 

    #    # Element-wise multiplication to scale each row of the matrix
       true_branch_tensor *= u_gripper_expand
 
       false_branch_tensor *= u_gripper_expand

       u_fingers = torch.where(u_gripper >= 0.0,
                           true_branch_tensor,
                           false_branch_tensor)

       self._finger_control[:, :] = u_fingers

       # Deploy actions
       self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control)) #gripper base
       self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control)) #gripper fingers


   def post_physics_step(self):
       
       self.progress_buf += 1

       env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
       if len(env_ids) > 0:
           self.reset_idx(env_ids)

       self.compute_observations()
       self.compute_reward(self.actions)

       # debug viz
       if self.viewer and self.debug_viz:
           self.gym.clear_lines(self.viewer)
           self.gym.refresh_rigid_body_state_tensor(self.sim)


           # Grab relevant states to visualize
           eef_pos = self.states["eef_pos"]
           eef_rot = self.states["eef_quat"]
           object_pos = self.states["object_pos"]
           object_rot = self.states["object_quat"]


           # Plot visualizations
           for i in range(self.num_envs):
               for pos, rot in zip((eef_pos, object_pos), (eef_rot, object_rot)):
                   px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                   py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                   pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()


                   p0 = pos[i].cpu().numpy()
                   self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                   self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                   self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_gripper_reward(
   reset_buf, progress_buf, successes, success_rate, xy_center, goal, quat, actions, states, reward_settings, max_episode_length
):
   # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
   # # Compute per-env physical parameters
   # object_size = states["object_size"]
   
   
   # Reset condition
   object_off_table = states["object_pos"][:, 2] < 0.2
   off_table = (states["eef_f1_pos"][:,2] < reward_settings["table_height"])\
               | (abs(states["eef_f1_pos"][:,1]) > 1.0) | (abs(states["eef_f1_pos"][:,0]) > 1.0) 
                   # | (states["eef_pos"][:,2] > 1.8)
      
   # reward for lifting object
   object_height = states["object_pos"][:, 2] - reward_settings["table_height"]
   object_lifted = (object_height) > 0.1
   lift_reward = object_lifted * reward_settings["r_lift_scale"]
    
   # reward for minimizing distance from gripper to the object
   d = torch.norm(states["object_pos_relative"], dim=-1)
   d_f1 = torch.norm(states["object_pos"] - states["eef_f1_pos"], dim=-1)
   d_f2 = torch.norm(states["object_pos"] - states["eef_f2_pos"], dim=-1)
   d_f3 = torch.norm(states["object_pos"] - states["eef_f3_pos"], dim=-1)
   
   
   dist_reward = (1 - torch.tanh(10.0 * (d + d_f1 + d_f2 + d_f3) / 4))
   dist_reward *= reward_settings["r_dist_scale"]

   # reward for object to reach the goal
#    d_goal = torch.norm(states["object_pos"] - goal, dim=-1)
   d_goal = 1.5 - states["object_pos"][:, 2]
   goal_condition = d_goal >= 0


#    goal_reward = torch.where((d_goal < 0.05), torch.ones_like(d_goal), 0)
   goal_reward = (1-torch.tanh(10*d_goal)) * goal_condition
   goal_reward *= reward_settings["r_goal_scale"] 
   
   object_reach_goal = (d_goal < 0.15) 
   reach_goal_reward = (d_goal < 0.4)  
   
   # reward for fitting the object between fingers
   fit_in_finger = torch.norm((states["eef_f1_pos"]+states["eef_f2_pos"]+states["eef_f3_pos"])/3 - states["object_pos"], dim=-1)
   pos_object_reward = (1 - torch.tanh(10*(fit_in_finger)))*0.02

   # penalty for not staying vertical after grasping :
   quat_difference = torch.norm(states["eef_quat"] - quat)
   quat_difference_x = states["eef_quat"][:,0] - quat[0]
   quat_difference_y = states["eef_quat"][:,1] - quat[1]
   quat_difference_z = quat[2] - states["eef_quat"][:,2]
   quat_difference_w = quat[3] - states["eef_quat"][:,3] 
   
   quat_x_reward = (1-torch.tanh(10*quat_difference_x)) * (quat_difference_x>0)
   quat_y_reward = (1-torch.tanh(10*quat_difference_y)) * (quat_difference_y>0)
   quat_z_reward = (1-torch.tanh(10*quat_difference_z)) * (quat_difference_z>0)
   quat_w_reward = (1-torch.tanh(10*quat_difference_w)) * (quat_difference_w>0)
   
   quat_reward = 1.0 * (quat_x_reward+quat_y_reward+quat_z_reward+quat_w_reward) * reach_goal_reward  * goal_condition
   
#    quat_reward = (1 - torch.tanh(10.0 * (quat_difference))) * 0.5
    
   # object height is not rewarded far away from center
   max_xy_drift = 0.2
   xy_drift = torch.norm(states["object_pos"][:, 0:2] - xy_center, dim=-1)
   close_to_center = xy_drift <= max_xy_drift
   lift_reward *= close_to_center.float()
   
   
   # Compose rewards
   rewards = (dist_reward) \
            + (lift_reward) \
            + goal_reward \
            + pos_object_reward \
            + quat_reward 
             
   # Compute resets
   reset_buf = torch.where((progress_buf >= max_episode_length - 1)| off_table | object_off_table,
                           torch.ones_like(reset_buf), reset_buf )
   
   # determine total number of successful episodes and success rate
   total_resets = torch.sum(reset_buf)
   avg_factor = 0.01
   successful_resets = torch.sum(object_reach_goal.float())
   successes += object_lifted.float()
   success_rate = torch.where(total_resets > 0,
                               avg_factor * (successful_resets/object_height.size(0)) +
                               (1. - avg_factor) * success_rate,
                               success_rate)

   return rewards, reset_buf, dist_reward, lift_reward, goal_reward, quat_reward, pos_object_reward, successes, success_rate
