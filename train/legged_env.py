import torch
import torch.nn.functional as F
import math
import genesis as gs
# from genesis.utils.terrain import parse_terrain

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import numpy as np
import random
import copy
from pathlib import Path
import tempfile, os, subprocess

URDF_ROOT = Path("/home/wataru/drl_ws/ts_Genesis/genesis/assets/urdf")
PLANE_DIR = URDF_ROOT / "plane"
XACRO     = PLANE_DIR / "custom_plane.urdf.xacro"

def compile_xacro_to_urdf(xacro_path: Path, out_path: Path | None = None, **mappings) -> str:
    xacro_path = Path(xacro_path)
    if not xacro_path.exists():
        raise FileNotFoundError(f"xacro not found: {xacro_path}")

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf")
        out_path = Path(tmp.name)
        tmp.close()
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # まず Python API を試す
    try:
        import xacro
        doc = xacro.process_file(str(xacro_path), mappings={k: str(v) for k, v in mappings.items()})
        out_path.write_text(doc.toxml(), encoding="utf-8")
        return str(out_path)
    except Exception as e:
        # CLI フォールバック（標準 xml を潰す 'xml/' ディレクトリがあると失敗するので注意）
        cmd = ["xacro", str(xacro_path)] + [f"{k}:={v}" for k, v in mappings.items()] + ["-o", str(out_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return str(out_path)
        except subprocess.CalledProcessError as ce:
            raise RuntimeError(f"xacro conversion failed:\n{ce.stderr or ce.stdout}") from e

def build_plane_urdf_with_step(step_h: float, out_path: str | Path = None) -> str:
    if step_h <= 0:
        raise ValueError("step_h must be > 0 (meters).")
    if out_path is None:
        out_path = PLANE_DIR / f"custom_plane_step{step_h:.3f}.urdf"
    return compile_xacro_to_urdf(XACRO, out_path=out_path, step_h_arg=step_h)

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# Helper function to get quaternion from Euler angles
def quaternion_from_euler_tensor(roll_deg, pitch_deg, yaw_deg):
    """
    roll_deg, pitch_deg, yaw_deg: (N,) PyTorch tensors of angles in degrees.
    Returns a (N, 4) PyTorch tensor of quaternions in [x, y, z, w] format.
    """
    # Convert to radians
    roll_rad = torch.deg2rad(roll_deg)
    pitch_rad = torch.deg2rad(pitch_deg)
    yaw_rad = torch.deg2rad(yaw_deg)

    # Half angles
    half_r = roll_rad * 0.5
    half_p = pitch_rad * 0.5
    half_y = yaw_rad * 0.5

    # Precompute sines/cosines
    cr = half_r.cos()
    sr = half_r.sin()
    cp = half_p.cos()
    sp = half_p.sin()
    cy = half_y.cos()
    sy = half_y.sin()

    # Quaternion formula (XYZW)
    # Note: This is the standard euler->quat for 'xyz' rotation convention.
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    # Stack into (N,4)
    return torch.stack([qw, qx, qy, qz], dim=-1)


def get_height_at_xy(height_field, x, y, horizontal_scale, vertical_scale, center_x, center_y):
    # Convert world coordinates to heightfield indices
    mat = np.array([[0, 1/horizontal_scale],
                    [1/horizontal_scale, 0]])
    vec = np.array([x+center_x, y+center_y])
    result = mat @ vec
    i = int(result[1])
    j = int(result[0])
    if 0 <= i < height_field.shape[0] and 0 <= j < height_field.shape[1]:
        return height_field[i, j] * vertical_scale
    else:
        raise ValueError(f"Requested (x={x}, y={y}) is outside the terrain bounds.")

def build_obs_buf(component_data_dict, obs_components): # e.x.) obs_components: self.obs_components
    obs_buf = []
    for name in obs_components:
        obs_buf.append(component_data_dict[name])
    return torch.cat(obs_buf, dim=-1)

ANG_VEL_EPS = 1e-5  # Treat small values as zero
class LeggedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg, show_viewer_=False, eval_=False, show_camera_=False, control_=False, device="cuda"):
        self.cfg = {
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "noise_cfg": noise_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "terrain_cfg": terrain_cfg,
        }
        self.eval = eval_
        self.control_ = control_
        self.show_camera_ = show_camera_
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = obs_cfg["num_privileged_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.command_curriculum = command_cfg["curriculum"]
        self.curriculum_complete_flag = False
        self.current_iteration = 0
        self.curriculum_iteration_threshold = command_cfg["curriculum_iteration_threshold"]
        self.enable_stop_commands = bool(command_cfg.get("enable_stop_commands", False))
        self.stop_command_probability = float(command_cfg.get("stop_command_probability", 0.0))
        self.stop_command_probability = min(max(self.stop_command_probability, 0.0), 1.0)
        # self.joint_limits = env_cfg["joint_limits"]
        self.simulate_action_latency = env_cfg["simulate_action_latency"]  # there is a 1 step latency on real robot
        self.dt = 1 / env_cfg['control_freq']
        self.sim_dt = self.dt / env_cfg['decimation']
        self.sim_substeps = 1
        self.low_level_cfg = env_cfg.get("low_level_control", {})
        self.use_low_level_control = self.low_level_cfg.get("enabled", False)
        self.enable_first_order_hold = self.low_level_cfg.get("first_order_hold", False)
        self.low_level_cutoff_hz = self.low_level_cfg.get("cutoff_freq_hz", 37.5)
        self.low_level_max_vel = self.low_level_cfg.get("max_joint_velocity", None)
        if self.low_level_max_vel is not None:
            self.low_level_max_vel = float(self.low_level_max_vel)
            if self.low_level_max_vel <= 0.0:
                self.low_level_max_vel = None
        self.low_level_alpha = None
        self._foh_factors = None
        if self.use_low_level_control:
            self._initialize_low_level_control(env_cfg)
        self.mean_reward = 0
        self.max_episode_length_s = env_cfg['episode_length_s']
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.noise_cfg = noise_cfg
        self.terrain_cfg = terrain_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.clip_obs = obs_cfg["clip_observations"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.noise_scales = noise_cfg["noise_scales"]
        self.selected_terrains = terrain_cfg["selected_terrains"]

        self.obs_components = self.obs_cfg["obs_components"]
        self.privileged_obs_components = self.obs_cfg["privileged_obs_components"]
        # Height-patch sensing (height map slice around the base)
        hp_cfg = env_cfg.get("height_patch", {})
        self.height_patch_enabled = bool(hp_cfg.get("enabled", False))
        self.height_patch_size = float(hp_cfg.get("size_m", 1.0))
        self.height_patch_points = int(hp_cfg.get("grid_points", 10))
        self.height_patch_res = self.height_patch_size / max(1, self.height_patch_points - 1)
        self.height_patch_dim = self.height_patch_points * self.height_patch_points
        hp_lin = torch.linspace(
            -0.5 * self.height_patch_size,
            0.5 * self.height_patch_size,
            self.height_patch_points,
            device=self.device,
            dtype=gs.tc_float,
        )
        hp_x, hp_y = torch.meshgrid(hp_lin, hp_lin, indexing="ij")
        self.height_patch_offsets = torch.stack((hp_x, hp_y), dim=-1)  # (G, G, 2)

        self.merged_components = []
        for comp in self.obs_components + self.privileged_obs_components:
            if comp not in self.merged_components:
                self.merged_components.append(comp)
        print("merged_components: ", self.merged_components)

        self.mirror_func_dict = None
        if self.env_cfg["mirror"]:
            import genesis.utils.mirror as mirror
            self.mirror_func_dict = {
                k: getattr(mirror, v) for k, v in self.obs_cfg["mirror_func"].items()
            }
            print(self.mirror_func_dict)

        self.component_dim_dict = None
        
        # if self.env_cfg["randomize_delay"]:
        # 1️⃣ Define Delay Parameters
        self.min_delay, self.max_delay = self.env_cfg["delay_range"]  # Delay range in seconds
        self.max_delay_steps = int(self.max_delay / self.dt)  # Convert max delay to steps

        # 2️⃣ Initialize Delay Buffers
        self.action_delay_buffer = torch.zeros(
            (self.num_envs, self.num_actions, self.max_delay_steps + 1), device=self.device
        )
        self.motor_delay_steps = torch.randint(
            int(self.min_delay / self.dt), self.max_delay_steps + 1,
            (self.num_envs, self.num_actions), device=self.device
        )
        # create scene
        self.mean_reward_threshold = self.command_cfg["mean_reward_threshold"]
        self.terrain_type = terrain_cfg["terrain_type"]
        visualized_number = min(num_envs, 100)        # capped at 100
        if self.terrain_type == "plane" and visualized_number > 3:
            visualized_number = 3
        elif self.terrain_type == "custom_plane" and visualized_number > 3:
            visualized_number = 20

        self.rendered_envs_idx = list(range(visualized_number))  

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=self.sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.env_cfg['decimation']),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=self.rendered_envs_idx),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=env_cfg['self_collision'],
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer_,
        )


        self.show_vis = (self.show_camera_ or show_viewer_)
        self.selected_robot = 0
        if self.show_camera_:
            self.cam_0 = self.scene.add_camera(
                res=(640, 480),
                pos=(5.0, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=self.show_camera_        
            )
        else:
            self.cam_0 = self.scene.add_camera(
                # res=(640, 480),
                pos=(5.0, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=self.show_camera_        
            )
        # Frame/video recording is optional; keep it off unless explicitly enabled in env_cfg.
        self.enable_recording = bool(env_cfg.get("enable_recording", False))
        self._recording = False
        self.mean_reward_flag = False
        self.mean_reward_half_flag = False
        self._recorded_frames = []
        self.available_positions = []

        subterrain_size = terrain_cfg["subterrain_size"]
        horizontal_scale = terrain_cfg["horizontal_scale"]
        vertical_scale = terrain_cfg["vertical_scale"]        
        if self.terrain_type != "plane" and self.terrain_type != "custom_plane" and self.terrain_type != "single_step":
            # # add plain

            ########################## entities ##########################
            self.cols = terrain_cfg["cols"]
            self.rows = terrain_cfg["rows"]
            n_subterrains=(self.cols, self.rows)
            terrain_types = list(self.selected_terrains.keys())
            probs = [terrain["probability"] for terrain in self.selected_terrains.values()]
            total = sum(probs)
            normalized_probs = [p / total for p in probs]
            subterrain_grid  = self.generate_subterrain_grid(self.rows, self.cols, terrain_types, normalized_probs)


            # Calculate the total width and height of the terrain
            total_width = (self.cols)* subterrain_size
            total_height =(self.rows)* subterrain_size

            # Calculate the center coordinates
            self.center_x = total_width / 2
            self.center_y = total_height / 2

            self.terrain  = gs.morphs.Terrain(
                pos=(-self.center_x,-self.center_y,0),
                subterrain_size=(subterrain_size, subterrain_size),
                n_subterrains=n_subterrains,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                subterrain_types=subterrain_grid,
            )        


            self.terrain_min_x = - (total_width  / 2.0)
            self.terrain_max_x =   (total_width  / 2.0)
            self.terrain_min_y = - (total_height / 2.0)
            self.terrain_max_y =   (total_height / 2.0)
            # Calculate the center of each subterrain in world coordinates

            self.global_terrain = self.scene.add_entity(self.terrain)
            
            self.subterrain_centers = []
            terrain_origin_x, terrain_origin_y, terrain_origin_z = self.terrain.pos
            self.height_field = self.global_terrain.geoms[0].metadata["height_field"]

            for row in range(self.rows):
                for col in range(self.cols):
                    subterrain_center_x = terrain_origin_x + (col + 0.5) * subterrain_size
                    subterrain_center_y = terrain_origin_y + (row + 0.5) * subterrain_size
                    subterrain_center_z = get_height_at_xy(
                        self.height_field,
                        subterrain_center_x,
                        subterrain_center_y,
                        horizontal_scale,
                        vertical_scale,
                        self.center_x,
                        self.center_y
                    )
                    self.subterrain_centers.append(
                        (subterrain_center_x, subterrain_center_y, subterrain_center_z)
                    )
            # Goal targets: subterrain centers
            self.goal_positions = list(self.subterrain_centers)

        else:
            if self.terrain_type == "custom_plane":
                # self.scene.add_entity(
                #     gs.morphs.URDF(file="urdf/plane/custom_plane.urdf", fixed=True),
                # ) 
                step_height = terrain_cfg["step_height"]
                print(f"step_height is {step_height}")
                urdf_path = build_plane_urdf_with_step(step_height)  # 絶対パスを返す

                self.scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True))

                # Beam thickness in the URDF
                t = 0.20

                # Rings as (inner_radius_R, height_H). 2 m gap => R = 1, 3, 5, 7 (m)
                rings = [
                    (2.7125, 0.0),  # inner1(2.1) と middle(3.325) の間の中点
                    (4.6625, 0.0),  # middle(3.475) と stair1(5.85) の間の中点
                ]

                # Slight lift to avoid initial interpenetration with the mesh
                EPS = 0.02  # 2 cm

                self.available_positions = []

                # Center (on the ground)
                self.available_positions.append((0.0, 0.0, 0.0 + EPS))

                # For each ring, place 4 positions at the centers of top/bottom/left/right beams
                for R, H in rings:
                    offset = R + t / 2.0     # beam center from origin
                    z_top = H                # top surface (URDF set with origin at H/2)
                    z_spawn = z_top + EPS

                    # Cardinal points on the ring (x, y, z)
                    self.available_positions.extend([
                        ( 0.0,  offset, z_spawn),  # top beam center
                        ( 0.0, -offset, z_spawn),  # bottom beam center
                        ( offset,  0.0,  z_spawn), # right beam center
                        (-offset,  0.0,  z_spawn), # left beam center
                    ])
                self.goal_positions = list(self.available_positions)

            elif self.terrain_type == "single_step":
                self.scene.add_entity(
                    gs.morphs.URDF(file="urdf/plane/single_step.urdf", fixed=True),
                ) 

                # Beam thickness in the URDF
                t = 0.20

                # Rings as (inner_radius_R, height_H). 2 m gap => R = 1, 3, 5, 7 (m)
                rings = [
                    (0.5, 0.10),  # 10 cm high
                    (2.5, 0.17),  # 17 cm high
                    (4.5, 0.25),  # 25 cm high
                    (6.5, 0.30),  # 35 cm high
                ]

                # Slight lift to avoid initial interpenetration with the mesh
                EPS = 0.02  # 2 cm

                self.available_positions = []

                # Center (on the ground)
                self.available_positions.append((0.0, 0.0, 0.0 + EPS))

                for R, H in rings:
                    offset = R + t / 2.0     # beam center from origin
                    z_top = H                # top surface (URDF set with origin at H/2)
                    z_spawn = z_top + EPS

                    # Cardinal points on the ring (x, y, z)
                    self.available_positions.extend([
                        ( 0.0,  offset, z_spawn),  # top beam center
                        ( 0.0, -offset, z_spawn),  # bottom beam center
                        ( offset,  0.0,  z_spawn), # right beam center
                        (-offset,  0.0,  z_spawn), # left beam center
                    ])
                self.goal_positions = list(self.available_positions)
            else:
                self.scene.add_entity(
                    gs.morphs.Plane(),
                )
                self.available_positions = []

                # Center (on the ground)
                self.available_positions.append((0.0, 0.0, 0.0))
                self.goal_positions = list(self.available_positions)

                # for R, H in rings:
                #     offset = R + t / 2.0     # beam center from origin
                #     z_top = H                # top surface (URDF set with origin at H/2)
                #     z_spawn = z_top + EPS

                #     # Cardinal points on the ring (x, y, z)
                #     self.available_positions.extend([
                #         ( 0.0,  offset, 0.0),  # top beam center
                #         ( 0.0, -offset, 0.0),  # bottom beam center
                #         ( offset,  0.0,  0.0), # right beam center
                #         (-offset,  0.0,  0.0), # left beam center
                #     ])
            # self.random_pos = self.generate_positions()
            # self.available_positions.append((0.0, 0.0, 0.0))
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # Goal pool (subterrain centers or fallback to spawn positions)
        if not hasattr(self, "goal_positions"):
            self.goal_positions = list(self.available_positions)
        self.goal_pool = list(self.goal_positions)
        self.goal_pool_len = len(self.goal_pool)
        self.goal_used_mask = None  # (N_envs, goal_pool_len) set in init_buffers
        if self.env_cfg["use_mjcf"]:
            self.robot  = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=self.env_cfg["robot_description"],
                    pos=self.base_init_pos.cpu().numpy(),
                    quat=self.base_init_quat.cpu().numpy(),
                ),
                visualize_contact=False,
            )
        else:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=self.env_cfg["robot_description"],
                    merge_fixed_links=True,
                    links_to_keep=self.env_cfg['links_to_keep'],
                    pos=self.base_init_pos.cpu().numpy(),
                    quat=self.base_init_quat.cpu().numpy(),
                ),
            )

        # build
        self.scene.build(n_envs=num_envs)
        if self.terrain_type != "plane" and self.terrain_type != "custom_plane" and self.terrain_type != "single_step":
            # use dense grid for respawn sampling
            step = 1.0  # 1m 間隔
            terrain_origin_x, terrain_origin_y, terrain_origin_z = self.terrain.pos
            self.height_field = self.global_terrain.geoms[0].metadata["height_field"]

            x_vals = np.arange(self.terrain_min_x, self.terrain_max_x, step)
            y_vals = np.arange(self.terrain_min_y, self.terrain_max_y, step)

            for x in x_vals:
                for y in y_vals:
                    try:
                        z = get_height_at_xy(
                            self.height_field,
                            x,
                            y,
                            horizontal_scale,
                            vertical_scale,
                            self.center_x,
                            self.center_y
                        )
                        self.available_positions.append((x, y, z))
                    except ValueError:
                        # 範囲外は無視
                        continue

            print(f"Stored {len(self.available_positions)} positions for respawn; {len(self.goal_positions)} goal centers")

            # self.subterrain_centers = []
            # # Get the terrain's origin position in world coordinates
            # terrain_origin_x, terrain_origin_y, terrain_origin_z = self.terrain.pos
            # self.height_field = self.global_terrain.geoms[0].metadata["height_field"]
            # for row in range(self.rows):
            #     for col in range(self.cols):
            #         subterrain_center_x = terrain_origin_x + (col + 0.5) * subterrain_size
            #         subterrain_center_y = terrain_origin_y + (row + 0.5) * subterrain_size
            #         # subterrain_center_z = (self.height_field[int(subterrain_center_x), int(subterrain_center_y)] ) * vertical_scale 
            #         subterrain_center_z = get_height_at_xy(
            #             self.height_field,
            #             subterrain_center_x,
            #             subterrain_center_y,
            #             horizontal_scale,
            #             vertical_scale,
            #             self.center_x,
            #             self.center_y
            #         )
                    
            #         print(f"Height at ({subterrain_center_x},{subterrain_center_y}): {subterrain_center_z}")
            #         self.subterrain_centers.append((subterrain_center_x, subterrain_center_y, subterrain_center_z))

            # Print the centers
            self.spawn_counter = 0
            # self.max_num_centers = len(self.subterrain_centers)
            # self.random_pos = self.generate_random_positions()
            self.height_field_tensor = torch.tensor(
                self.height_field, device=self.device, dtype=gs.tc_float
            )
        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_start for name in self.env_cfg["dof_names"]]
        self.hip_dofs = [self.robot.get_joint(name).dof_start for name in self.env_cfg["hip_joint_names"]]
        self.thigh_dofs = [self.robot.get_joint(name).dof_start for name in self.env_cfg["thigh_joint_names"]]
        def find_link_indices(names):
            link_indices = list()
            for name in names:
                for link in self.robot.links:
                    if name in link.name:
                        link_indices.append(link.idx - self.robot.link_start)
            return link_indices


        self.termination_contact_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        )
        self.penalised_contact_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        )
        self.calf_indices = find_link_indices(
            self.env_cfg['calf_link_name']
        )

        self.feet_indices = find_link_indices(
            self.env_cfg['feet_link_name']
        )
        self.non_foot_indices = np.setdiff1d(
            np.arange(self.robot.n_links, dtype=np.int64),
            np.array(self.feet_indices, dtype=np.int64)
        )
        self.num_feet = len(self.feet_indices)  # 2 for biped, 4 for quad, etc.
        self.base_link_index = find_link_indices(
            self.env_cfg['base_link_name']
        )
        self.thigh_indices = find_link_indices(
            self.env_cfg['thigh_link_name']
        )
        self.head_indices = find_link_indices(
            "head"
        )
        self.body_half_length = torch.tensor(
            self.env_cfg.get("body_half_length", 0.35),  # 例: 長手方向の半分[m]
            device=self.device, dtype=gs.tc_float
        )
        self.body_half_width = torch.tensor(
            self.env_cfg.get("body_half_width", 0.16),   # 例: 幅方向の半分[m]
            device=self.device, dtype=gs.tc_float
        )

        def _map_calf_to_foot_indices():
            calf_to_foot = {}
            valid_prefixes = ["FL", "FR", "RL", "RR"]

            calf_names = self.env_cfg['calf_link_name']          # list like ["calf"]
            foot_suffix = self.env_cfg['feet_link_name'][0]      # e.g., "foot"

            for calf_link in self.robot.links:
                if any(name in calf_link.name for name in calf_names):
                    prefix = calf_link.name.split("_")[0]  # e.g., "FL"
                    if prefix not in valid_prefixes:
                        continue

                    target_foot_name = f"{prefix}_{foot_suffix}"
                    for foot_link in self.robot.links:
                        if foot_link.name == target_foot_name:
                            calf_idx = calf_link.idx - self.robot.link_start
                            foot_idx = self.feet_indices[valid_prefixes.index(prefix)]
                            calf_to_foot[calf_idx] = foot_idx

            return calf_to_foot

        self.calf_to_foot_map = _map_calf_to_foot_indices()
        print(f"calf to foot map {self.calf_to_foot_map}")
        print(f"motor dofs {self.motor_dofs}")
        print(f"feet indicies {self.feet_indices}")
        relative_lower_height_threshold = self.env_cfg["termination_if_relative_height_lower_than"]
        print(f"termination relative lower height {relative_lower_height_threshold}")
        # PD control
        stiffness = self.env_cfg['PD_stiffness']
        damping = self.env_cfg['PD_damping']
        force_limit = self.env_cfg['force_limit']

        self.p_gains, self.d_gains, self.force_limits = [], [], []
        for dof_name in self.env_cfg['dof_names']:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        for dof_name in self.env_cfg['dof_names']:
            for key in force_limit.keys():
                if key in dof_name:
                    self.force_limits.append(force_limit[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)
        # Set the force range using the calculated force limits
        self.robot.set_dofs_force_range(
            lower=-np.array(self.force_limits),  # Negative lower limit
            upper=np.array(self.force_limits),   # Positive upper limit
            dofs_idx_local=self.motor_dofs
        )
        # Store link indices that trigger termination or penalty
        self.feet_front_indices = self.feet_indices[:2]
        self.feet_rear_indices = self.feet_indices[2:]

        self.termination_exceed_degree_ignored = False
        self.termination_if_roll_greater_than_value = self.env_cfg["termination_if_roll_greater_than"]
        self.termination_if_pitch_greater_than_value = self.env_cfg["termination_if_pitch_greater_than"]
        if self.termination_if_roll_greater_than_value <= 1e-6 or self.termination_if_pitch_greater_than_value <= 1e-6:
            self.termination_exceed_degree_ignored = True

        print(f"termination exceed degree ignored is {self.termination_exceed_degree_ignored}")

        print("=== Link index → name mapping (enumerate _links) ===")
        for idx, link in enumerate(self.robot._links):
            # RigidLink usually has .name; if not, dir(link) to inspect
            print(f"{idx:3d} : {getattr(link, 'name', '<no name>')}")

        print("Base link idx:", self.robot.base_link_idx)
        
        print(f"termination_contact_indicies {self.termination_contact_indices}")
        print(f"penalised_contact_indices {self.penalised_contact_indices}")
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            if name=="termination":
                continue
            self.reward_functions[name] = getattr(self, "_reward_" + name)

        # initialize buffers
        self.init_buffers()
        print("Done initializing")



    def _initialize_low_level_control(self, env_cfg):
        decimation = max(1, int(env_cfg.get("decimation", 1)))
        if self.enable_first_order_hold:
            self._foh_factors = torch.linspace(
                1, decimation, steps=decimation, device=self.device, dtype=gs.tc_float
            ) / decimation
        else:
            self._foh_factors = torch.ones(
                decimation, device=self.device, dtype=gs.tc_float
            )
        cutoff = self.low_level_cutoff_hz
        if cutoff is None or cutoff <= 0.0:
            self.low_level_alpha = 1.0
        else:
            rc = 1.0 / (2.0 * math.pi * cutoff)
            self.low_level_alpha = float(self.sim_dt / (rc + self.sim_dt))

    def _apply_low_pass_filter(self, desired_action):
        """
        desired_action : (N, n_dof)   方策出力（通常は [-1,1] スケール）
        戻り値         : (N, n_dof)   フィルタ後の方策出力
        方式:
        1) 行動 → 関節角目標 q_ref_des [rad] に変換
        2) q_ref 空間でレート制限（max_joint_velocity [rad/s]）
        3) q_ref 空間で一次IIRローパス（EMA）
        4) 行動空間に戻す
        """
        if not self.use_low_level_control:
            return desired_action

        # 1) 行動 → 関節角目標 [rad]
        #    action_scale はスカラー/配列のどちらでもOKにしてブロードキャスト
        if torch.is_tensor(self.env_cfg['action_scale']):
            action_scale = self.env_cfg['action_scale'].to(self.device, dtype=gs.tc_float)
        else:
            action_scale = torch.as_tensor(self.env_cfg['action_scale'], device=self.device, dtype=gs.tc_float)

        # shape を (N, n_dof) に合わせてブロードキャスト
        if action_scale.ndim == 0:
            action_scale = action_scale.expand_as(desired_action)
        elif action_scale.shape != desired_action.shape[-1:]:
            # (n_dof,) → (N, n_dof)
            action_scale = action_scale.unsqueeze(0).expand_as(desired_action)

        q_ref_des = self.default_dof_pos + desired_action * action_scale + self.motor_offsets  # [rad]

        # 2) フィルタ内部状態（q_ref [rad]）の初期化
        if getattr(self, "low_level_filter_state", None) is None:
            self.low_level_filter_state = q_ref_des.clone()
            # 行動空間に戻して返す
            return (self.low_level_filter_state - self.default_dof_pos - self.motor_offsets) / action_scale

        # 3) レート制限：q_ref 空間で [rad/s] を [rad/step] に
        if self.low_level_max_vel is not None and self.low_level_max_vel > 0.0:
            # max_joint_velocity はスカラー/配列の両対応
            v_max = torch.as_tensor(self.low_level_max_vel, device=self.device, dtype=gs.tc_float)
            if v_max.ndim == 0:
                v_max = v_max.expand_as(q_ref_des)
            elif v_max.shape != q_ref_des.shape[-1:]:
                v_max = v_max.unsqueeze(0).expand_as(q_ref_des)

            dq_max = v_max * self.sim_dt  # [rad/step]
            dq     = torch.clamp(q_ref_des - self.low_level_filter_state, min=-dq_max, max=dq_max)
            q_limited = self.low_level_filter_state + dq
        else:
            q_limited = q_ref_des

        # 4) 一次IIRローパス（EMA）：q_ref 空間
        alpha = float(self.low_level_alpha)
        if alpha >= 1.0:
            self.low_level_filter_state.copy_(q_limited)
        else:
            self.low_level_filter_state.add_(alpha * (q_limited - self.low_level_filter_state))

        # 5) 行動空間に戻す
        action_filtered = (self.low_level_filter_state - self.default_dof_pos - self.motor_offsets) / action_scale
        return action_filtered


    def _apply_motor_torques(self, motor_actions):
        self.torques = self._compute_torques(motor_actions)
        if self.num_envs == 0:
            torques = self.torques.squeeze()
            self.robot.control_dofs_force(torques, self.motor_dofs)
        else:
            self.robot.control_dofs_force(self.torques, self.motor_dofs)
        self.scene.step()
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

    def init_buffers(self):
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )

        
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.zero_obs = torch.zeros(self.num_obs, device=self.device, dtype=gs.tc_float)
        self.zero_privileged_obs = torch.zeros(self.num_privileged_obs, device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        self.registered_yaw_buf = torch.zeros(self.num_envs, device=self.device,dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.time_out_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.out_of_bounds_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["base_lin_vel"], self.obs_scales["base_lin_vel"], self.obs_scales["base_ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.hip_actions = torch.zeros((self.num_envs, len(self.hip_dofs)), device=self.device, dtype=gs.tc_float)
        self.thigh_actions = torch.zeros((self.num_envs, len(self.thigh_dofs)), device=self.device, dtype=gs.tc_float)

        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.idle_leg_raise_duration = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_int,
        )

        self.episode_returns = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.noise_scale_vec = None if self.component_dim_dict is None else self._get_noise_scale_vec()
        print("noise scale vector: ", self.noise_scale_vec)
        self.actions = torch.zeros_like(self.actions)
        self.last_actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_prev_actions = torch.zeros_like(self.actions)
        if self.use_low_level_control:
            self.prev_low_level_target = None
            self.low_level_filter_state = None
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.hip_pos = torch.zeros_like(self.hip_actions)
        self.hip_vel = torch.zeros_like(self.hip_actions)
        self.thigh_dof_pos = torch.zeros_like(self.thigh_actions)
        self.thigh_dof_vel = torch.zeros_like(self.thigh_actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        # --- ゴール追従用バッファ ----------------------------------
        self.goal_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )  # 世界座標でのゴール位置
        self.has_goal = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )  # ゴールを持っているかどうか
        self.goal_reached_flag = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )  # ゴール到達を1 stepだけ通知するフラグ
        if self.goal_pool_len > 0:
            self.goal_used_mask = torch.zeros(
                (self.num_envs, self.goal_pool_len), device=self.device, dtype=torch.bool
            )
        self.goal_max_speed = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_float
        )  # 各ゴールに対して設定する最大速度
        self.goal_mode = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )  # 0:なし, 1:正面, 2:左側面, 3:右側面, 4:後ろ向き
        # ------------------------------------------------------------        
        
        if self.height_patch_enabled:
            hp_shape = (self.num_envs, self.height_patch_points, self.height_patch_points)
            self.height_patch_world = torch.zeros(hp_shape, device=self.device, dtype=gs.tc_float)
            self.height_patch_rel = torch.zeros(hp_shape, device=self.device, dtype=gs.tc_float)
            self.height_patch_flat = torch.zeros(self.num_envs, self.height_patch_dim, device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.num_dof = len(self.default_dof_pos )
        self.default_hip_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
                if name in self.env_cfg["hip_joint_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.default_thigh_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
                if name in self.env_cfg["thigh_joint_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.contact_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        self.leg_cross_duration_buf = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.swing_stuck_ema = torch.zeros(
            self.num_envs,
            dtype=gs.tc_float,
            device=self.device,
            requires_grad=False,
        )
        self.low_height_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        self.pitch_exceed_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device
        )
        self.roll_exceed_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device
        )
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # Iterate over the motor DOFs
        # self.soft_dof_vel_limit = self.env_cfg["soft_dof_vel_limit"]
        # ❶  Prefer user-supplied arrays if they exist
        if "dof_lower_limit" in self.env_cfg and "dof_upper_limit" in self.env_cfg:
            lower = torch.tensor(
                self.env_cfg["dof_lower_limit"], device=self.device, dtype=gs.tc_float
            )
            upper = torch.tensor(
                self.env_cfg["dof_upper_limit"], device=self.device, dtype=gs.tc_float
            )

            if lower.shape[0] != len(self.motor_dofs) or upper.shape[0] != len(self.motor_dofs):
                raise ValueError(
                    f"dof_lower/upper_limit lengths ({lower.shape[0]}/{upper.shape[0]}) "
                    f"must match #motor_dofs ({len(self.motor_dofs)})."
                )

            # stack → shape (n_dof, 2)  column-0 = lower, column-1 = upper
            self.dof_pos_limits = torch.stack([lower, upper], dim=1)

        else:
            # ❷  Fallback: read limits from the robot model
            #     (unchanged behaviour)
            self.dof_pos_limits = torch.stack(
                self.robot.get_dofs_limit(self.motor_dofs), dim=1
            )
        # ❸  Torque limits are still read from the model (unchanged)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        soft_factor = self.reward_cfg["soft_dof_pos_limit"]  # e.g. 0.9  → 90 % range
        self.soft_torque_limit = self.reward_cfg["soft_torque_limit"]  # e.g. 0.9  → 90 % range
        centres = self.dof_pos_limits.mean(dim=1)            # (n_dof,)
        ranges  = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]

        self.dof_pos_limits[:, 0] = centres - 0.5 * ranges * soft_factor
        self.dof_pos_limits[:, 1] = centres + 0.5 * ranges * soft_factor

        self.motor_strengths = gs.ones((self.num_envs, self.num_dof), dtype=float)
        self.motor_offsets = gs.zeros((self.num_envs, self.num_dof), dtype=float)
        self.link_friction = torch.ones(
            (self.num_envs, self.robot.n_links), device=self.device, dtype=gs.tc_float
        )

        # 足リンクだけの摩擦を保持 (N, F)  F = #feet
        self.foot_friction = torch.ones(
            (self.num_envs, self.num_feet), device=self.device, dtype=gs.tc_float
        )
        self.init_foot()
        # self._randomize_controls()
        # self._randomize_rigids()
        self.num_legs = self.num_feet             # Go2なら 4
        self.joints_per_leg = self.num_dof // self.num_legs
        if self.num_dof % self.num_legs != 0:
            raise ValueError("num_dof が脚数で割り切れません")

        self.torque_ema = torch.zeros(
            (self.num_envs, self.num_legs, self.joints_per_leg),
            device=self.device,
            dtype=gs.tc_float,
        )
        print(f"Dof_pos_limits{self.dof_pos_limits}")
        print(f"Default dof pos {self.default_dof_pos}")
        print(f"Default hip pos {self.default_hip_pos}")
        self.front_hip_idx = torch.tensor([0, 1], device=self.device)  # FR_hip, FL_hip
        self.rear_hip_idx  = torch.tensor([2, 3], device=self.device)  # RR_hip, RL_hip
        self.common_step_counter = 0
        # extras
        # self.continuous_push = torch.zeros(
        #     (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        # )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        self.prev_rel_h = torch.zeros((self.num_envs,), device=self.device)
        # Mapping from actual link index (e.g., 13) → index in feet_pos[:, :, :]
        self.link_idx_to_feet_tensor_idx = {
            link_idx: i for i, link_idx in enumerate(self.feet_indices)
        }

        # 重力の大きさ（取得できなければ env_cfg から/既定 9.81）
        try:
            gvec = torch.tensor(self.scene.sim.rigid_solver.gravity, device=self.device, dtype=gs.tc_float)
            self.g_mag = gvec.norm()  # or: self.g_mag = gvec.abs()[2]
        except Exception:
            self.g_mag = torch.tensor(self.env_cfg.get("gravity_mag", 9.81), device=self.device, dtype=gs.tc_float)

        # 名目リンク質量（全リンクの初期値を足し合わせ）
        self.nominal_link_masses = torch.tensor(
            [lk.get_mass() for lk in self.robot.links], device=self.device, dtype=gs.tc_float
        )  # (L,)
        self.nominal_total_mass = self.nominal_link_masses.sum()      # スカラー

        # 各リンクの mass_shift を覚えておく（ランダム化で更新）
        self.mass_shift_buf = torch.zeros(
            (self.num_envs, len(self.robot.links)), device=self.device, dtype=gs.tc_float
        )  # (N,L)

        # env ごとの総質量（ランダム化を反映）
        self.total_mass_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        # 初期化時に一度リフレッシュ
        self._refresh_total_mass()

        self.base_obs_components = [
            c for c in self.obs_components
            if c not in ("ema_fast", "ema_slow")
        ]
        self.base_obs_dim = None  # 最初の compute_observations で決める

        self.obs_ema_fast = None  # (num_envs, base_obs_dim)
        self.obs_ema_slow = None

    def get_terrain_height_at(self, x_world, y_world):
        if self.terrain_cfg["terrain_type"] == "plane" or self.terrain_cfg["terrain_type"] == "custom_plane" or self.terrain_cfg["terrain_type"] == "single_step":
            return torch.zeros_like(x_world, device=self.device)
    
        s = self.terrain_cfg["horizontal_scale"]
    
        # Convert scalar inputs to tensors
        x_world = torch.as_tensor(x_world, device=self.device)
        y_world = torch.as_tensor(y_world, device=self.device)
    
        x_shifted = x_world + self.center_x
        y_shifted = y_world + self.center_y
    
        i = (y_shifted / s).long().clamp(0, self.height_field_tensor.shape[0] - 1)
        j = (x_shifted / s).long().clamp(0, self.height_field_tensor.shape[1] - 1)
    
        return self.height_field_tensor[i, j] * self.terrain_cfg["vertical_scale"]

    def get_terrain_height_at_for_base(self, x_world, y_world):
        if self.terrain_cfg["terrain_type"] == "plane" or self.terrain_cfg["terrain_type"] == "custom_plane" or self.terrain_cfg["terrain_type"] == "single_step":
          return torch.tensor(0.0, device=self.device)
        # Create the transform matrix (same as your NumPy one)
        s = self.terrain_cfg["horizontal_scale"]
        mat = torch.tensor([[0.0, 1.0 / s],
                            [1.0 / s, 0.0]], device=self.device)

        # Shift world position by center
        vec = torch.stack([x_world + self.center_x,
                        y_world + self.center_y])

        # Apply transformation
        result = mat @ vec

        i = result[1].long().clamp(0, self.height_field_tensor.shape[0] - 1)
        j = result[0].long().clamp(0, self.height_field_tensor.shape[1] - 1)

        return self.height_field_tensor[i, j] * self.terrain_cfg["vertical_scale"]


    def assign_commands_euler(self, envs_idx):
        # range を取り出し
        ang_min, ang_max = self.command_cfg["ang_vel_range"]

        # min == max （-0.0 も 0.0 と等しい）なら return
        if abs(ang_min) <=1e-6 and abs(ang_max) <= 1e-6:
            return

        self.commands[envs_idx, 0] = 0.0
        self.commands[envs_idx, 1] = 0.0
        self.commands[envs_idx, 2] = gs_rand_float(
            ang_min, ang_max, (len(envs_idx),), self.device
        )
        self._maybe_apply_stop_commands(envs_idx)


    def assign_fixed_commands(self, envs_idx):
        """
        Assign one of four randomly-sampled fixed-direction commands to each
        environment index in `envs_idx`, based on (env_id % 4):
            0 → forward  (+x, random between [0.5, max])
            1 → backward (−x, random between [min, -0.5])
            2 → right    (+y,  random between [0.5, max])
            3 → left     (−y,  random between [min, -0.5])
        """
        envs_idx = torch.as_tensor(envs_idx, device=self.device, dtype=torch.long)
        n_envs = len(envs_idx)
        n_cmds = 4

        cmd_types = (envs_idx % n_cmds).long()
        cmds = torch.zeros((n_envs, 3), device=self.device)

        # Masks for command types
        fwd_mask   = cmd_types == 0
        bwd_mask   = cmd_types == 1
        right_mask = cmd_types == 2
        left_mask  = cmd_types == 3

        # Forward (+x)
        cmds[fwd_mask, 0] = gs_rand_float(0.1, self.command_cfg["lin_vel_x_range"][1], (fwd_mask.sum(),), self.device)

        # Backward (-x)
        cmds[bwd_mask, 0] = gs_rand_float(self.command_cfg["lin_vel_x_range"][0], -0.1, (bwd_mask.sum(),), self.device)

        # Right (+y)
        cmds[right_mask, 1] = gs_rand_float(0.1, self.command_cfg["lin_vel_y_range"][1], (right_mask.sum(),), self.device)

        # Left (-y)
        cmds[left_mask, 1] = gs_rand_float(self.command_cfg["lin_vel_y_range"][0], -0.1, (left_mask.sum(),), self.device)

        # Apply to global buffer
        self.commands[envs_idx] = cmds
        # self._current_command_types[envs_idx] = cmd_types  # Track types only for updated envs

        self._maybe_apply_stop_commands(envs_idx)
        

    def biased_sample(self, min_val, max_val, size, device, bias=2.0):
        """
        Sample values with bias towards positive range.
        The bias parameter skews values towards the upper end.
        """
        uniform_samples = torch.rand(size, device=device)  # [0, 1] uniform
        skewed_samples = uniform_samples ** (1.0 / bias)  # Biasing towards 1
        return min_val + (max_val - min_val) * skewed_samples


    def _resample_commands(self, envs_idx):
        if True:
            self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        else:
            self.commands[envs_idx, 0] = self.biased_sample(
                *self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device, bias=2.0
            )
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        self._maybe_apply_stop_commands(envs_idx)

    def _zero_commands(self, envs_idx):
        self.commands[envs_idx, 0] = 0.0
        self.commands[envs_idx, 1] = 0.0
        self.commands[envs_idx, 2] = 0.0

    def _maybe_apply_stop_commands(self, envs_idx):
        """
        Optionally zero commands for a random subset of envs.
        Controlled via command_cfg fields:
          - enable_stop_commands (bool)
          - stop_command_probability (float in [0,1])
        """
        if not self.enable_stop_commands or self.stop_command_probability <= 0.0:
            return
        envs_idx = torch.as_tensor(envs_idx, device=self.device, dtype=torch.long)
        if envs_idx.numel() == 0:
            return
        stop_mask = torch.rand(envs_idx.numel(), device=self.device) < self.stop_command_probability
        if stop_mask.any():
            self._zero_commands(envs_idx[stop_mask])


    def generate_subterrain_grid(self, rows, cols, terrain_types, weights):
        """
        Generate a 2D grid (rows x cols) of terrain strings chosen randomly
        based on 'weights', but do NOT place 'pyramid_sloped_terrain' adjacent 
        to another 'pyramid_sloped_terrain'.
        """
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        max_attempts = 10

        def has_adjacent_match(r, c, predicate):
            # Only need to check already-filled neighbors (up/left) while building row-wise
            for dr, dc in [(-1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor = grid[nr][nc]
                    if neighbor is not None and predicate(neighbor):
                        return True
            return False

        def resolve_choice(raw_choice):
            """Map raw weighted choice to a final terrain token."""
            if raw_choice == "pyramid_sloped_terrain":
                return random.choice(["pyramid_sloped_terrain", "pyramid_down_sloped_terrain"])
            if raw_choice == "pyramid_stairs_terrain":
                terrain_options = ["pyramid_stairs_terrain", "pyramid_down_stairs_terrain"]
                terrain_weights = [0.0, 1.0]  # climb up priority
                return random.choices(terrain_options, weights=terrain_weights, k=1)[0]
            return raw_choice

        for i in range(rows):
            for j in range(cols):
                terrain_choice = None
                for _ in range(max_attempts):
                    candidate = resolve_choice(random.choices(terrain_types, weights=weights, k=1)[0])

                    # Avoid adjacent slopes and stairs
                    if "stair" in candidate and has_adjacent_match(i, j, lambda t: "stair" in t):
                        continue

                    terrain_choice = candidate
                    break

                # Fallback: choose any non-conflicting terrain if repeated attempts failed
                if terrain_choice is None:
                    non_conflicting = []
                    for raw in terrain_types:
                        candidate = resolve_choice(raw)
                        if "stair" in candidate and has_adjacent_match(i, j, lambda t: "stair" in t):
                            continue
                        non_conflicting.append(candidate)
                    terrain_choice = random.choice(non_conflicting) if non_conflicting else resolve_choice(terrain_types[0])

                grid[i][j] = terrain_choice
        return grid

    def init_foot(self):
        self.feet_num = len(self.feet_indices)
       
        self.step_period = self.reward_cfg["step_period"]
        self.step_offset = self.reward_cfg["step_offset"]
        self.step_height_for_front = self.reward_cfg["front_feet_relative_height"]
        self.step_height_for_rear = self.reward_cfg["rear_feet_relative_height"]
        #todo get he first feet_pos here
        # Get positions for all links and slice using indices
        all_links_pos = self.robot.get_links_pos()
        all_links_vel = self.robot.get_links_vel()

        self.feet_pos = all_links_pos[:, self.feet_indices, :]
        self.thigh_pos =  all_links_pos[:, self.thigh_indices, :]
        self.feet_front_pos = all_links_pos[:, self.feet_front_indices, :]
        self.feet_rear_pos = all_links_pos[:, self.feet_rear_indices, :]
        self.feet_vel = all_links_vel[:, self.feet_indices, :]
        self.calf_pos = all_links_pos[:, self.calf_indices, :]
        self.calf_vel = all_links_vel[:, self.calf_indices, :]
        self.front_feet_pos_base = self._world_to_base_transform(self.feet_front_pos, self.base_pos, self.base_quat)
        self.rear_feet_pos_base = self._world_to_base_transform(self.feet_rear_pos, self.base_pos, self.base_quat)
        self.thigh_pos_base =  self._world_to_base_transform(self.thigh_pos, self.base_pos, self.base_quat)

    def update_feet_state(self):
        # Get positions for all links and slice using indices
        all_links_pos = self.robot.get_links_pos()
        all_links_vel = self.robot.get_links_vel()

        self.feet_pos = all_links_pos[:, self.feet_indices, :]
        self.thigh_pos =  all_links_pos[:, self.thigh_indices, :]
        self.calf_pos = all_links_pos[:, self.calf_indices, :]
        self.calf_vel = all_links_vel[:, self.calf_indices, :]
        self.feet_front_pos = all_links_pos[:, self.feet_front_indices, :]
        self.feet_rear_pos = all_links_pos[:, self.feet_rear_indices, :]
        self.feet_vel = all_links_vel[:, self.feet_indices, :]
        self.front_feet_pos_base = self._world_to_base_transform(self.feet_front_pos, self.base_pos, self.base_quat)
        self.rear_feet_pos_base = self._world_to_base_transform(self.feet_rear_pos, self.base_pos, self.base_quat)
        self.thigh_pos_base =  self._world_to_base_transform(self.thigh_pos, self.base_pos, self.base_quat)


    def _quaternion_to_matrix(self, quat):
        w, x, y, z = quat.unbind(dim=-1)
        R = torch.stack([
            1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
        ], dim=-1).reshape(-1, 3, 3)
        return R

    def _world_to_base_transform(self, points_world, base_pos, base_quat):
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_matrix(base_quat)

        # Subtract base position to get relative position
        points_relative = points_world - base_pos.unsqueeze(1)

        # Apply rotation to transform to base frame
        points_base = torch.einsum('bij,bkj->bki', R.transpose(1, 2), points_relative)
        return points_base

    def _update_height_patch(self):
        if not self.height_patch_enabled:
            return
        base_xy = self.base_pos[:, :2]  # (N,2)
        base_z = self.base_pos[:, 2].view(self.num_envs, 1, 1)  # (N,1,1)

        # Broadcast offsets
        world_x = base_xy[:, None, None, 0] + self.height_patch_offsets[..., 0]
        world_y = base_xy[:, None, None, 1] + self.height_patch_offsets[..., 1]

        if self.terrain_type in ("plane", "custom_plane", "single_step"):
            heights_world = torch.zeros_like(world_x, device=self.device, dtype=gs.tc_float)
        else:
            hf = self.height_field_tensor  # shape (rows, cols)
            hs = self.terrain_cfg["horizontal_scale"]
            vs = self.terrain_cfg["vertical_scale"]

            idx_x = torch.floor((world_x + self.center_x) / hs).long()
            idx_y = torch.floor((world_y + self.center_y) / hs).long()
            idx_x = torch.clamp(idx_x, 0, hf.shape[0] - 1)
            idx_y = torch.clamp(idx_y, 0, hf.shape[1] - 1)
            heights_world = hf[idx_x, idx_y] * vs

        self.height_patch_world[:] = heights_world
        self.height_patch_rel[:] = heights_world - base_z
        self.height_patch_flat[:] = self.height_patch_rel.view(self.num_envs, -1)

    def _get_feet_forces_and_contact(self, force_thresh=1.0):
        """
        returns:
        feet_forces: [N, 4, 3]  各足の接触力ベクトル (x,y,z)
        contact_mask: [N, 4]    True/False（または float で 0/1）
        前後の index を結合して「足順」を固定（例: [FR, FL, RR, RL]）にする。
        """
        # 例: それぞれ 2本ずつ入っている想定
        # self.feet_front_indices = (FR_idx, FL_idx)
        # self.feet_rear_indices  = (RR_idx, RL_idx)
        feet_indices = torch.tensor(
            [*self.feet_front_indices, *self.feet_rear_indices],
            dtype=torch.long, device=self.contact_forces.device
        )

        # 力ベクトルをそのまま抜き出す（ノルムは取らない）
        feet_forces = self.contact_forces[:, feet_indices, :3]  # [N, 4, 3]

        # 接地判定はノルムで（スカラー）
        contact_mask = (torch.norm(feet_forces, dim=2) > force_thresh)  # [N, 4]
        # 計算に掛けたいなら float にする
        contact_mask_f = contact_mask.float()

        return feet_forces, contact_mask, contact_mask_f

    def post_physics_step_callback(self):
        self.update_feet_state()
        self.phase = (self.episode_length_buf * self.dt) % self.step_period / self.step_period
        # Assign phases for quadruped legs
        """
        small_offset = 0.05  # tweak as needed, 0 < small_offset < step_offset typically
        self.phase_FL_RR = self.phase
        self.phase_FR_RL = (self.phase + self.step_offset) % 1

        # Now offset one leg in each diagonal pair slightly
        phase_FL = self.phase_FL_RR
        phase_RR = (self.phase_FL_RR + small_offset) % 1     # shifted by small_offset

        phase_FR = self.phase_FR_RL
        phase_RL = (self.phase_FR_RL + small_offset) % 1     # shifted by small_offset

        # Concatenate in the order (FL, FR, RL, RR)
        self.leg_phase = torch.cat([
            phase_FL.unsqueeze(1),
            phase_FR.unsqueeze(1),
            phase_RL.unsqueeze(1),
            phase_RR.unsqueeze(1)
        ], dim=-1)feet_rear_indices
        """
        if self.show_vis:
            self._draw_debug_vis()
        # Assign phases for quadruped legs
        if self.num_feet == 4:
            # Quadruped: FL/RR in phase, FR/RL offset by step_offset
            phase_FL_RR = self.phase                                      # (N,)
            phase_FR_RL = (self.phase + self.step_offset) % 1.0           # (N,)
            self.leg_phase = torch.cat([
                phase_FL_RR.unsqueeze(1),  # FL
                phase_FR_RL.unsqueeze(1),  # FR
                phase_FR_RL.unsqueeze(1),  # RL
                phase_FL_RR.unsqueeze(1),  # RR
            ], dim=1)                                                      # (N,4)
        elif self.num_feet == 2:
            # Biped: left/right 180°*step_offset apart
            left_phase  = self.phase                                       # (N,)
            right_phase = (self.phase + self.step_offset) % 1.0            # (N,)
            self.leg_phase = torch.cat([left_phase.unsqueeze(1),
                                        right_phase.unsqueeze(1)], dim=1)  # (N,2)
        else:
            # Generic: evenly spaced offsets
            offsets = torch.linspace(0, 1, steps=self.num_feet+1, device=self.device)[:-1]  # (F,)
            self.leg_phase = (self.phase.unsqueeze(1) + offsets) % 1.0      # (N,F)
        


    def step(self, actions):
        # if self.episode_length_buf >0:
        # 新しい actions を使う前に履歴をずらす
        self.prev_prev_actions[:] = self.prev_actions
        self.prev_actions[:] = self.last_actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        if self.env_cfg["randomize_delay"] and self.mean_reward_flag:
            # 3️⃣ Store new actions in delay buffer (Shift the buffer)
            self.action_delay_buffer[:, :, :-1] = self.action_delay_buffer[:, :, 1:].clone()
            self.action_delay_buffer[:, :, -1] = self.actions  # Insert latest action

            # 3) Vectorized gather for delayed actions

            T = self.action_delay_buffer.shape[-1]  # T = max_delay_steps + 1
            # (num_envs, num_actions)
            delayed_indices = (T - 1) - self.motor_delay_steps
            # Expand to (num_envs, num_actions, 1)
            gather_indices = delayed_indices.unsqueeze(-1)

            # Gather from last dimension
            delayed_actions = self.action_delay_buffer.gather(dim=2, index=gather_indices).squeeze(-1)

            commanded_actions = delayed_actions
        else:
            commanded_actions = self.last_actions if self.simulate_action_latency else self.actions

        decimation = max(1, int(self.env_cfg['decimation']))
        if self.use_low_level_control:
            prev_target = getattr(self, "prev_low_level_target", None)
            if prev_target is None:
                prev_target = torch.zeros_like(commanded_actions)
                self.prev_low_level_target = prev_target
            action_delta = commanded_actions - prev_target
            for substep_idx in range(decimation):
                if self.enable_first_order_hold and decimation > 1:
                    ratio = self._foh_factors[substep_idx]
                    hold_action = prev_target + ratio * action_delta
                else:
                    hold_action = commanded_actions
                filtered_action = self._apply_low_pass_filter(hold_action)
                self._apply_motor_torques(filtered_action)
            self.prev_low_level_target = commanded_actions.clone()
        else:
            for _ in range(decimation):
                self._apply_motor_torques(commanded_actions)


        # --- トルク EMA 更新 --------------------------------
        torques_leg = self.torques.view(self.num_envs, self.num_legs, self.joints_per_leg)
        alpha_prev = self.reward_cfg.get("effort_ema_alpha", 0.975)  # 過去の重み
        self.torque_ema = alpha_prev * self.torque_ema + (1.0 - alpha_prev) * torques_leg
        # -----------------------------------------------------


        pos_after_step = self.robot.get_pos()
        quat_after_step = self.robot.get_quat()
        # base_height = pos_after_step[:, 2]      # (N,)
        # base_height_vals = base_height.detach().cpu().flatten().tolist()
        # if len(base_height_vals) == 1:
        #     print(f"Base height: {base_height_vals[0]:.4f} [m]")
        # else:
        #     heights_str = ", ".join(f"{val:.4f}" for val in base_height_vals)
        #     print(f"Base heights: [{heights_str}] [m]")
        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.hip_pos[:] = self.robot.get_dofs_position(self.hip_dofs)
        self.hip_vel[:] = self.robot.get_dofs_velocity(self.hip_dofs)
        self.thigh_dof_pos[:] = self.robot.get_dofs_position(self.thigh_dofs)
        self.thigh_dof_vel[:] = self.robot.get_dofs_velocity(self.thigh_dofs)
        self.contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )        
        if self.height_patch_enabled:
            self._update_height_patch()
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        half_episode_idx = (
            (self.episode_length_buf >= int(self.max_episode_length * 0.5))
            .nonzero(as_tuple=False)
            .flatten()
        )

        #resample commands here
        
        # Boolean mask: True where command includes non-zero angular velocity
        has_ang_cmd = torch.abs(self.commands[:, 2]) > ANG_VEL_EPS
        # Indices of environments WITH angular velocity command
        # Indices of environments WITHOUT angular velocity command
        envs_idx_no_ang_cmd = (~has_ang_cmd).nonzero(as_tuple=False).flatten()
        if self.command_curriculum and not self.curriculum_complete_flag:
            if self.mean_reward_flag:
                self.curriculum_complete_flag = True
            elif not self.control_:
                both_mask = torch.isin(envs_idx, half_episode_idx)
                envs_half_and_resample = envs_idx[both_mask]
                # Then apply Euler command assignment only to those that also have no ang command
                envs_half_and_resample_no_ang = envs_half_and_resample[
                    ~has_ang_cmd[envs_half_and_resample]
                ]
                self.assign_fixed_commands(envs_idx)
                self.assign_commands_euler(envs_half_and_resample_no_ang)
        elif not self.control_:
            self._resample_commands(envs_idx)

        # --- ここでゴール付き env の commands を毎ステップ上書き ---
        self._assign_goal_commands_every_step()
        # ---------------------------------------------------------

        envs_idx_with_ang_cmd = (
            torch.abs(self.commands[:, 2]) > ANG_VEL_EPS
        ).nonzero(as_tuple=False).flatten()    # 1‑D LongTensor of env indices
        self.registered_yaw_buf[envs_idx_with_ang_cmd] = self.base_euler[envs_idx_with_ang_cmd, 2]
        self.post_physics_step_callback()
        
        # random push
        self.common_step_counter += 1
        push_interval_s = self.env_cfg['push_interval_s']
        if push_interval_s > 0:
            max_push_vel_xy = self.env_cfg['max_push_vel_xy']
            dofs_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)



        self.check_termination()
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # compute reward
        self.compute_rewards()

        self.compute_observations()

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        self._render_headless()
        self.extras["observations"]["critic"] = self.privileged_obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)
        cos_phase = torch.cos(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)

        # Prepare all components
        base_lin_vel = self.base_lin_vel * self.obs_scales["base_lin_vel"]
        base_ang_vel = self.base_ang_vel * self.obs_scales["base_ang_vel"]
        projected_gravity = self.projected_gravity
        commands = self.commands * self.commands_scale
        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos_scaled"]
        dof_vel_scaled = self.dof_vel * self.obs_scales["dof_vel_scaled"]
        torques_scaled = self.torques * self.obs_scales["torques_scaled"]
        actions = self.actions
        # ─── post_physics_step_callback() ────────────────────────────────
        swing_mask = (self.leg_phase > 0.55)                 # (N,4)
        foot_force = torch.norm(self.contact_forces[:,       # (N,4)
                                self.feet_indices, :3], dim=2)

        collision  = (swing_mask & (foot_force > 1.0)).float()   # (N,4)
        foot_mu = self.foot_friction
        # Debug checks
        debug_items = {
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "commands": commands,
            "dof_pos_scaled": dof_pos_scaled,
            "dof_vel_scaled": dof_vel_scaled,
            "actions": actions,
        }

        # まず「生のコンポーネント」を辞書に集める
        component_data_dict = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "commands": commands,
            "dof_pos_scaled": dof_pos_scaled,
            "dof_vel_scaled": dof_vel_scaled,
            "torques_scaled": torques_scaled,
            "actions": actions,
            "sin_phase": sin_phase,
            "cos_phase": cos_phase,
            "collision": collision,
            "foot_friction": foot_mu,
        }
        if self.height_patch_enabled:
            component_data_dict["height_patch"] = self.height_patch_flat

        # --- 1) base_obs を構成（EMA対象） --------------------
        base_obs = build_obs_buf(component_data_dict, self.base_obs_components)
        if self.base_obs_dim is None:
            self.base_obs_dim = base_obs.shape[-1]

        # --- 2) EMA を更新 -----------------------------------
        if self.obs_ema_fast is None:
            self.obs_ema_fast = base_obs.clone()
            self.obs_ema_slow = base_obs.clone()
        else:
            alpha_f = self.obs_cfg.get("ema_alpha_fast", 0.5)
            alpha_s = self.obs_cfg.get("ema_alpha_slow", 0.87)
            self.obs_ema_fast = alpha_f * self.obs_ema_fast  + (1 - alpha_f) * base_obs
            self.obs_ema_slow = alpha_s * self.obs_ema_slow  + (1 - alpha_s) * base_obs

        # --- 3) EMA を component_data_dict に追加 -------------
        component_data_dict["ema_fast"] = self.obs_ema_fast
        component_data_dict["ema_slow"] = self.obs_ema_slow

        # --- 4) component_dim_dict の初期化 -------------------
        if self.component_dim_dict is None:
            self.component_dim_dict = {}
            for key, value in component_data_dict.items():
                self.component_dim_dict[key] = value.shape[-1]
            self.noise_scale_vec = self._get_noise_scale_vec()
            print("noise scale vector: ", self.noise_scale_vec)

        # --- 5) obs_buf / privileged_obs_buf を構成 -----------
        self.obs_buf = build_obs_buf(component_data_dict, self.obs_components)
        self.privileged_obs_buf = build_obs_buf(component_data_dict, self.privileged_obs_components)

        if self.obs_buf.shape[1] != self.num_obs:
            raise ValueError(f"obs_buf dim {self.obs_buf.shape[1]} does not match configured num_obs={self.num_obs}")
        if self.privileged_obs_buf.shape[1] != self.num_privileged_obs:
            raise ValueError(
                f"privileged_obs_buf dim {self.privileged_obs_buf.shape[1]} "
                f"does not match configured num_privileged_obs={self.num_privileged_obs}"
            )

        self.obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -self.clip_obs, self.clip_obs)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if torch.isnan(self.obs_buf).any() or torch.isinf(self.obs_buf).any():
            print(">>> WARNING: NaN or Inf in final obs_buf <<<")



    def compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.reward_cfg["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        self.episode_returns += self.rew_buf     # add current step reward

    def unpack_obs(self, obs, obs_components):
        """
        flat な obs を辞書に分解する。
        obs, components
         = self.obs_buf, self.obs_components
         or self.privileged_obs_buf, self.privileged_obs_components
        """
        obs_dict = {}
        start_idx = 0
        for key in obs_components:
            value = self.component_dim_dict[key]
            obs_dict[key] = obs[..., start_idx:start_idx+value]
            start_idx += value
        return obs_dict

    def mirror_observation(self, obs, obs_components):
        obs_clone = obs.clone()
        obs_dict = self.unpack_obs(obs_clone, obs_components)
        for key in obs_components:
            mirror_func = self.mirror_func_dict[key]
            obs_dict[key] = mirror_func(obs_dict[key])
        return build_obs_buf(obs_dict, obs_components)

    def get_observations(self):
        self.extras["observations"]["critic"] = self.privileged_obs_buf

        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf



    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.noise_cfg["add_noise"]
        noise_level =self.noise_cfg["noise_level"]

        noise_dict = self.unpack_obs(noise_vec, self.obs_components)
        
        for key in self.obs_components:
            if key not in self.noise_scales:
                self.noise_scales[key] = 0.0

            if key not in self.obs_scales:
                self.obs_scales[key] = 1.0

            noise_dict[key] += self.noise_scales[key]*noise_level*self.obs_scales[key]

        return build_obs_buf(noise_dict, self.obs_components)


    def check_termination(self):
        """Check if environments need to be reset."""
        # (n_envs, n_links, 3) tensor of net contact forces
        # --- contact-based termination -------------------------------------------
        in_contact = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1
        )
        self.contact_duration_buf[in_contact] += self.dt
        self.contact_duration_buf[~in_contact] = 0.0
        reset_contact = self.contact_duration_buf > self.env_cfg["termination_duration"]

        # --- height-based termination --------------------------------------------
        terrain_h = self.get_terrain_height_at_for_base(self.base_pos[:, 0], self.base_pos[:, 1])
        rel_h = self.base_pos[:, 2] - terrain_h
        low_h = rel_h < self.env_cfg["termination_if_relative_height_lower_than"]
        self.low_height_duration_buf[low_h] += self.dt
        self.low_height_duration_buf[~low_h] = 0.0
        reset_low_h = self.low_height_duration_buf > self.env_cfg["termination_duration"]

        # --- lateral leg crossing guard (kinematic, no self-collision) -----------
        cross_term_depth = float(self.reward_cfg.get("cross_terminate_depth", 0.0))
        cross_term_duration = float(self.reward_cfg.get("cross_terminate_duration", 0.0))
        if cross_term_depth > 0.0:
            _, cross_depth = self._compute_leg_clearance_penalty(include_rear=True, return_depth=True)
            cross_violation = cross_depth > cross_term_depth
            self.leg_cross_duration_buf[cross_violation] += self.dt
            self.leg_cross_duration_buf[~cross_violation] = 0.0
            if cross_term_duration <= 0.0:
                reset_cross = cross_violation
            else:
                reset_cross = self.leg_cross_duration_buf > cross_term_duration
            safety_extras = self.extras.setdefault("safety", {})
            safety_extras["leg_cross_depth"] = cross_depth
            safety_extras["leg_cross_violation"] = cross_violation.float()
        else:
            reset_cross = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        # --- combine --------------------------------------------------------------
        self.reset_buf = reset_contact | reset_low_h | reset_cross

        if not self.termination_exceed_degree_ignored:
            # Check where pitch and roll exceed thresholds
            pitch_exceeded = torch.abs(self.base_euler[:, 1]) > math.radians(self.termination_if_pitch_greater_than_value)
            roll_exceeded = torch.abs(self.base_euler[:, 0]) > math.radians(self.termination_if_roll_greater_than_value)

            # Increment duration where exceeded
            self.pitch_exceed_duration_buf[pitch_exceeded] += self.dt
            self.roll_exceed_duration_buf[roll_exceeded] += self.dt

            # Reset duration where NOT exceeded
            self.pitch_exceed_duration_buf[~pitch_exceeded] = 0.0
            self.roll_exceed_duration_buf[~roll_exceeded] = 0.0

            # Trigger reset if exceed duration > threshold (e.g., 3 seconds)
            pitch_timeout = self.pitch_exceed_duration_buf > self.env_cfg["angle_termination_duration"]
            roll_timeout = self.roll_exceed_duration_buf > self.env_cfg["angle_termination_duration"]

            self.reset_buf |= pitch_timeout
            self.reset_buf |= roll_timeout

        if self.command_curriculum and not self.curriculum_complete_flag:
            yaw_limit_rad = math.radians(60)
            curr_yaw = self.base_euler[:, 2]
            delta_yaw = (curr_yaw - self.registered_yaw_buf + math.pi) % (2 * math.pi) - math.pi

            # Only consider environments that have NO angular velocity command

            # Apply yaw limit condition only to those
            exceed_yaw = (torch.abs(delta_yaw) > yaw_limit_rad)

            # if exceed_yaw.any():
            #     print("Kill some envs because they exceed the yaw limit without angular command")

            #     # Print the delta yaw in degrees for those that exceed the limit
            #     delta_deg = torch.rad2deg(delta_yaw)
            #     print("Delta Yaw (deg):", delta_deg[exceed_yaw])

            self.reset_buf |= exceed_yaw
        MAX_BODY_SPEED = 5.0  # [m/s]
        high_speed = torch.norm(self.base_lin_vel, dim=1) > MAX_BODY_SPEED 
        self.reset_buf |= high_speed
        # # shape (num_envs, num_dof) → Bool where True = violation
        # out_of_limits = (self.dof_pos < self.dof_pos_limits[:, 0]) | \
        #                 (self.dof_pos > self.dof_pos_limits[:, 1])

        # # any() along the dof dimension ⇒ shape (num_envs,)
        # joint_violation = out_of_limits.any(dim=1)

        # self.reset_buf |= joint_violation        # mark those envs for reset
        #wataru 

        # -------------------------------------------------------
        #  Add out-of-bounds check using terrain_min_x, etc.
        # -------------------------------------------------------
        # min_x, max_x, min_y, max_y = self.terrain_bounds  # or however you store them
        
        # We assume base_pos[:, 0] is x, base_pos[:, 1] is y
        if self.terrain_type != "plane" and self.terrain_type != "custom_plane" and self.terrain_type != "single_step":
            self.out_of_bounds_buf = (
                (self.base_pos[:, 0] < self.terrain_min_x) |
                (self.base_pos[:, 0] > self.terrain_max_x) |
                (self.base_pos[:, 1] < self.terrain_min_y) |
                (self.base_pos[:, 1] > self.terrain_max_y)
            )
            self.reset_buf |= self.out_of_bounds_buf
        # For those that are out of bounds, penalize by marking episode_length_buf = max
        # self.episode_length_buf[out_of_bounds] = self.max_episode_length


        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf



    def _assign_goal_commands_every_step(self):
        """
        has_goal == True の env だけ、
        ・ゴールへ向かうような線形速度コマンド
        ・ゴールに正対 / 側面 / 後ろ向き となる yaw コマンド
        を毎ステップ上書きする。
        """
        if not hasattr(self, "has_goal") or not self.has_goal.any():
            return

        goal_mask = self.has_goal
        idx = goal_mask.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return

        base_pos = self.base_pos[idx]      # (M,3)
        goal_pos = self.goal_pos[idx]      # (M,3)
        delta_xy = goal_pos[:, :2] - base_pos[:, :2]
        dist_xy = torch.norm(delta_xy, dim=1)  # (M,)

        # ゴールとほぼ同じ場所にいるやつはスキップ
        eps = 1e-3
        active_mask = dist_xy > eps
        if not active_mask.any():
            return

        idx = idx[active_mask]
        delta_xy = delta_xy[active_mask]
        dist_xy = dist_xy[active_mask]
        modes = self.goal_mode[idx]

        # --- yaw 方向の制御 ----------------------------------------
        # ゴール方向（世界座標系）: atan2(dy, dx)
        target_yaw = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])  # (K,)
        curr_yaw = self.base_euler[idx, 2]                        # (K,)

        # 「正面をゴールに向けたとき」の yaw
        yaw_facing_goal = target_yaw

        # モードに応じてオフセット追加
        yaw_offset = torch.zeros_like(yaw_facing_goal)
        yaw_offset[modes == 2] =  math.pi / 2.0   # 左側面をゴールに向ける
        yaw_offset[modes == 3] = -math.pi / 2.0   # 右側面
        yaw_offset[modes == 4] =  math.pi         # 後ろ向きでゴール側を向く

        yaw_des = yaw_facing_goal + yaw_offset
        # [-pi, pi] に wrap
        yaw_error = yaw_des - curr_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        k_yaw = self.command_cfg.get("goal_yaw_kp", 2.0)
        ang_min, ang_max = self.command_cfg["ang_vel_range"]
        yaw_cmd = torch.clamp(k_yaw * yaw_error, min=ang_min, max=ang_max)

        # --- 線形速度の制御 ----------------------------------------
        # ゴールまでの距離に比例してスピードを決める（上限あり）
        k_v = self.command_cfg.get("goal_v_kp", 0.5)
        max_v_x = self.command_cfg["lin_vel_x_range"][1]
        max_v_y = self.command_cfg["lin_vel_y_range"][1]
        v_max = min(max_v_x, max_v_y)
        # ゴール専用にサンプリングした最大速度があればそれを使う
        if hasattr(self, "goal_max_speed") and self.goal_max_speed.numel() >= self.num_envs:
            v_max_env = self.goal_max_speed[idx]
        else:
            v_max_env = torch.full_like(dist_xy, v_max)
        # clamp supports tensor-tensor; min must be tensor as well
        zero_floor = torch.zeros_like(dist_xy)
        v_des = torch.clamp(k_v * dist_xy, min=zero_floor, max=v_max_env)

        # base frame でのコマンド [vx, vy, yaw]
        cmds = torch.zeros((idx.numel(), 3), device=self.device, dtype=gs.tc_float)

        # デフォルトは正面方向へ進む
        cmds[:, 0] = v_des

        # 後ろ向きモードは後退
        cmds[modes == 4, 0] = -v_des[modes == 4]

        # 左右側面モードでは横移動のみでゴールを狙う（x=0, y 方向で最短移動）
        side_mask = (modes == 2) | (modes == 3)
        cmds[side_mask, 0] = 0.0
        cmds[modes == 2, 1] = -v_des[modes == 2]  # 左面を向く→右へステップ
        cmds[modes == 3, 1] =  v_des[modes == 3]  # 右面を向く→左へステップ

        # yaw コマンド
        cmds[:, 2] = yaw_cmd

        # 最後に self.commands を上書き
        self.commands[idx] = cmds

        # --- ゴール到達判定（一定距離以内に近づいたらゴール更新） ----
        goal_radius = self.command_cfg.get("goal_radius", 0.3)
        reached = dist_xy < goal_radius
        if reached.any():
            reached_envs = idx[reached]
            # ゴールに到達した env には新しいゴールを割り当てる
            self.goal_reached_flag[reached_envs] = True
            self._assign_new_goals(reached_envs)


    def _assign_new_goals(self, envs_idx, p_goal=None):
        """
        envs_idx に対して一定確率でゴールを割り当てる。
        p_goal: 1.0 にすると「必ずゴールを持つ」。
        """
        if p_goal is None:
            p_goal = self.command_cfg.get("goal_probability", 0.3)

        if len(envs_idx) == 0:
            return

        envs_idx = torch.as_tensor(envs_idx, device=self.device, dtype=torch.long)

        # 利用可能なゴール候補がなければ何もしない
        goal_pool = getattr(self, "goal_positions", self.available_positions)
        if len(goal_pool) == 0 or p_goal <= 0.0:
            self.has_goal[envs_idx] = False
            return

        # ゴールを持たせるかどうかを確率的に決める
        rand = torch.rand(len(envs_idx), device=self.device)
        use_goal_mask = rand < p_goal

        envs_with_goal = envs_idx[use_goal_mask]
        envs_no_goal   = envs_idx[~use_goal_mask]

        # ゴールを持たない env はフラグを落とす
        if envs_no_goal.numel() > 0:
            self.has_goal[envs_no_goal] = False
            self.goal_mode[envs_no_goal] = 0

        if envs_with_goal.numel() == 0:
            return

        # ゴール候補から「現在位置に最も近い」ものを選ぶ
        goal_pool_tensor = torch.as_tensor(goal_pool, device=self.device, dtype=gs.tc_float)  # (M,3)
        base_pos = self.base_pos[envs_with_goal]  # (K,3)
        prev_goal = self.goal_pos[envs_with_goal]  # (K,3)
        used_mask = None
        if self.goal_used_mask is not None and self.goal_used_mask.shape[1] == goal_pool_tensor.shape[0]:
            used_mask = self.goal_used_mask[envs_with_goal]  # (K,M)

        # 距離計算（K,M）
        diff = goal_pool_tensor.unsqueeze(0) - base_pos.unsqueeze(1)
        dist_sq = torch.sum(diff * diff, dim=2)
        same_goal = torch.all(
            torch.isclose(goal_pool_tensor.unsqueeze(0), prev_goal.unsqueeze(1), atol=1e-5),
            dim=2,
        )
        dist_sq = dist_sq.masked_fill(same_goal, float("inf"))
        if used_mask is not None:
            dist_sq = dist_sq.masked_fill(used_mask, float("inf"))
            all_inf = torch.isinf(dist_sq).all(dim=1)
            if all_inf.any():
                # reset usage for exhausted envs and recompute distances
                self.goal_used_mask[envs_with_goal[all_inf]] = False
                dist_sq_re = torch.sum(diff[all_inf] * diff[all_inf], dim=2)
                dist_sq[all_inf] = dist_sq_re.masked_fill(same_goal[all_inf], float("inf"))
        nearest_idx = torch.argmin(dist_sq, dim=1)  # (K,)

        chosen_positions = goal_pool_tensor[nearest_idx]  # (K,3)
        self.goal_pos[envs_with_goal] = chosen_positions
        if used_mask is not None:
            self.goal_used_mask[envs_with_goal, nearest_idx] = True

        # 各 env ごとに最大速度をランダム設定（>=0.5）
        lin_min, lin_max = self.command_cfg["lin_vel_x_range"]
        lin_min = max(0.5, lin_min)
        max_speed = torch.empty(envs_with_goal.numel(), device=self.device, dtype=gs.tc_float).uniform_(lin_min, lin_max)
        self.goal_max_speed[envs_with_goal] = max_speed

        # 1:正面, 2:左側面, 3:右側面, 4:後ろ向き （前進を優先的に付与）
        mode_probs = torch.tensor(
            [0.7, 0.05, 0.05, 0.20], device=self.device, dtype=gs.tc_float
        )
        mode_probs = mode_probs / mode_probs.sum()  # safety normalize
        cdf = torch.cumsum(mode_probs, dim=0)
        u = torch.rand(envs_with_goal.numel(), device=self.device)
        modes = torch.bucketize(u, cdf).to(torch.long) + 1  # bucket → {0..3} → {1..4}
        self.goal_mode[envs_with_goal] = modes
        self.has_goal[envs_with_goal] = True


    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # indices that are **not** being reset
        all_idx     = torch.arange(self.num_envs, device=self.device)
        keep_mask   = ~torch.isin(all_idx, torch.as_tensor(envs_idx, device=self.device))
        active_idx  = all_idx[keep_mask]

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # reset base
        # Check if the new_base_pos contains any NaNs
        # Randomly choose positions from pre-generated random_pos for each environment
        self.sample_random_positions(envs_idx)
        # random_indices = torch.randint(0, self.num_envs, (len(envs_idx),), device=self.device)
        # self.base_pos[envs_idx] = self.random_pos[random_indices] + self.base_init_pos
            

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        # if 0 in env_idx:
        if not self.mean_reward_flag:
            self.mean_reward_flag = self.mean_reward > self.mean_reward_threshold
        if not self.mean_reward_half_flag:
            self.mean_reward_half_flag = self.mean_reward > self.mean_reward_threshold/2
        if self.command_curriculum and not self.curriculum_complete_flag:
            self.curriculum_complete_flag =  self.current_iteration > self.curriculum_iteration_threshold
        else:
            self.curriculum_complete_flag = True

        if self.env_cfg["randomize_rot"] and ((self.mean_reward_flag and self.curriculum_complete_flag) or self.eval) :
            # 1) Get random roll, pitch, yaw (in degrees) for each environment.
            
            roll = gs_rand_float(*self.env_cfg["roll_range"],  (len(envs_idx),), self.device)
            pitch = gs_rand_float(*self.env_cfg["pitch_range"], (len(envs_idx),), self.device)
            yaw = gs_rand_float(*self.env_cfg["yaw_range"],    (len(envs_idx),), self.device)

            # 2) Convert them all at once into a (N,4) quaternion tensor [x, y, z, w].
            quats_torch = quaternion_from_euler_tensor(roll, pitch, yaw)  # (N, 4)

            # 3) Move to CPU if needed and assign into self.base_quat in one shot
            #    (assuming self.base_quat is a numpy array of shape [num_envs, 4]).
            self.base_quat[envs_idx] = quats_torch
        else:
            self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)

        if self.env_cfg["randomize_delay"]:
            self.motor_delay_steps[envs_idx] = torch.randint(
                int(self.min_delay / self.dt),
                self.max_delay_steps + 1,
                (len(envs_idx), self.num_actions),
                device=self.device
            )

        # 1b. Check DOFs
        dof_pos = self.robot.get_dofs_position(self.motor_dofs)

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)
        # reset buffers
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.action_delay_buffer[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.hip_pos[envs_idx] = 0.0
        self.hip_vel[envs_idx] = 0.0
        self.thigh_dof_pos[envs_idx] = 0.0
        self.thigh_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = -(0.5/ self.dt)
        self.feet_air_time[envs_idx] = 0.0
        self.idle_leg_raise_duration[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.reset_buf[envs_idx] = True
        self.contact_duration_buf[envs_idx] = 0.0
        self.leg_cross_duration_buf[envs_idx] = 0.0
        self.swing_stuck_ema[envs_idx] = 0.0
        self.low_height_duration_buf[envs_idx] = 0.0
        self.episode_returns[envs_idx]  = 0.0
        self.obs_buf[envs_idx]                = 0.0
        self.privileged_obs_buf[envs_idx]     = 0.0
        self.rew_buf[envs_idx]                = 0.0
        self.time_out_buf[envs_idx]           = 0
        self.out_of_bounds_buf[envs_idx]      = 0
        self.last_contacts[envs_idx]          = 0
        self.contact_forces[envs_idx]         = 0.0
        self.pitch_exceed_duration_buf[envs_idx] = 0.0
        self.roll_exceed_duration_buf[envs_idx]  = 0.0
        self.registered_yaw_buf[envs_idx]   =   0.0


        # fill extras
        if self.mean_reward_half_flag:
            self._randomize_rigids(envs_idx)
            self._randomize_controls(envs_idx)
        if self.use_low_level_control:
            self._reset_low_level_buffers(envs_idx)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._zero_commands(envs_idx)

        # --- ゴール関連のリセットと再割り当て -----------------------
        # 一旦フラグをクリアしてから、新しいゴールを割り当てる
        self.has_goal[envs_idx] = False
        self.goal_pos[envs_idx] = 0.0
        self.goal_mode[envs_idx] = 0
        self.goal_reached_flag[envs_idx] = False
        if self.goal_used_mask is not None and self.goal_used_mask.shape[0] == self.num_envs:
            self.goal_used_mask[envs_idx] = False
        self.goal_max_speed[envs_idx] = 0.0

        # ここで p_goal の確率でゴールを持たせる
        # 例: goal_probability が未設定なら 0.3 (30%) とする
        self._assign_new_goals(envs_idx)
        # --------------------------------------------------------------
        if self.env_cfg['send_timeouts']:
            self.extras['time_outs'] = self.time_out_buf


    def _reset_low_level_buffers(self, envs_idx):
        envs_idx = torch.as_tensor(envs_idx, device=self.device, dtype=torch.long)
        if envs_idx.numel() == 0:
            return

        if self.prev_low_level_target is None:
            self.prev_low_level_target = torch.zeros_like(self.actions)
        else:
            self.prev_low_level_target[envs_idx] = 0.0

        if isinstance(self.motor_offsets, torch.Tensor):
            motor_offsets = self.motor_offsets
        else:
            motor_offsets = torch.as_tensor(
                self.motor_offsets, device=self.device, dtype=self.default_dof_pos.dtype
            )

        default_state = self.default_dof_pos.unsqueeze(0) + motor_offsets

        if self.low_level_filter_state is None:
            self.low_level_filter_state = default_state.clone()
        else:
            self.low_level_filter_state[envs_idx] = default_state[envs_idx]


    def sample_random_positions(self, envs_idx):
        """
        Sample fresh random positions (x,y,z) for the given env indices.
        """
        n = len(envs_idx)
        # pick n random positions (with replacement)
        choices = random.choices(self.available_positions, k=n)

        # convert to tensor
        pos = torch.tensor(choices, dtype=torch.float32, device=self.device)

        # add base_init_pos offset
        self.base_pos[envs_idx] = pos + self.base_init_pos

    def generate_random_positions(self):
        """
        Use the _random_robot_position() method to generate unique random positions
        for each environment.
        """
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.num_envs):
            x, y, z = self._random_robot_position()
            # positions[i] = torch.tensor([0, 0, z], device=self.device)
            positions[i] = torch.tensor([x, y, z], device=self.device)
        return positions

    def generate_positions(self):
        """
        Use the _random_robot_position() method to generate unique random positions
        for each environment.
        """
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.num_envs):
            positions[i] = torch.tensor([0, 0, 0], device=self.device)
        return positions

    def _random_robot_position(self):
        # 1. Sample random row, col(a subterrain)
        # 0.775 ~ l2_norm(0.7, 0.31)
        # go2_size_xy = 0.775
        # row = np.random.randint(int((self.rows * self.terrain.subterrain_size[0]-go2_size_xy)/self.terrain.horizontal_scale))
        # col = np.random.randint(int((self.cols * self.terrain.subterrain_size[1]-go2_size_xy)/self.terrain.horizontal_scale))
        center = self.subterrain_centers[self.spawn_counter]
        x, y, z = center[0], center[1], center[2]
        self.spawn_counter+= 1
        if self.spawn_counter == len(self.subterrain_centers):
            self.spawn_counter = 0
       
        return x, y, z


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        
        actions_scaled = actions * self.env_cfg['action_scale']
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
            - self.batched_d_gains * self.dof_vel
        )
        torques =  torques * self.motor_strengths

            # torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, self.privileged_obs_buf


    def _render_headless(self):
        should_record = self.enable_recording and self._recording and len(self._recorded_frames) < 150
        if should_record:
            x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
            self.cam_0.set_pose(pos=(x+5.0, y, z+5.5), lookat=(x, y, z+0.5))
            if self.show_camera_:
                self.cam_0.render(
                    rgb=True,
                )
            frame, _, _, _ = self.cam_0.render()
            self._recorded_frames.append(frame)
        elif self.show_camera_:
            x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
            self.cam_0.set_pose(pos=(x+5.0, y, z+5.5), lookat=(x, y, z+0.5))
            self.cam_0.render(
                rgb=True,
            )
    def get_recorded_frames(self):
        if not self.enable_recording:
            return None
        if len(self._recorded_frames) >=10:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None

    def start_recording(self, record_internal=True):
        if not self.enable_recording:
            return
        if self.show_camera_:
            return
        self._recorded_frames = []
        self._recording = True
        if record_internal:
            self._record_frames = True
        else:
            self.cam_0.start_recording()

    def stop_recording(self, save_path=None):
        if not self.enable_recording:
            return
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self.dt))
            self.cam_0.stop_recording(save_path, fps = int(1 / self.dt))

    # ------------ domain randomization----------------

    def _randomize_rigids(self, env_ids=None):


        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['motor_randomize_friction']:
            self._randomize_motor_link_friction(env_ids)
        if self.env_cfg['foot_randomize_friction']:
            self._randomize_foot_link_friction(env_ids)
        if self.env_cfg['randomize_base_mass']:
            self._randomize_base_mass(env_ids)
        if self.env_cfg['randomize_com_displacement']:
            self._randomize_com_displacement(env_ids)


    def _randomize_controls(self, env_ids=None):

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_motor_strength']:
            self._randomize_motor_strength(env_ids)
        if self.env_cfg['randomize_motor_offset']:
            self._randomize_motor_offset(env_ids)
        if self.env_cfg['randomize_kp_scale']:
            self._randomize_kp(env_ids)
        if self.env_cfg['randomize_kd_scale']:
            self._randomize_kd(env_ids)

    def _randomize_motor_link_friction(self, env_ids):
        """足以外のリンクにだけ摩擦をランダム適用"""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0 or self.non_foot_indices.size == 0:
            return

        mu_min, mu_max = self.env_cfg['motor_friction_range']
        L = self.non_foot_indices.size

        rand = mu_min + (mu_max - mu_min) * torch.rand(
            (len(env_ids), L), device=self.device, dtype=gs.tc_float
        )

        self.robot.set_friction_ratio(
            friction_ratio=rand,
            links_idx_local=np.asarray(self.non_foot_indices, dtype=np.int64),
            envs_idx=env_ids,
        )

        # link_friction の該当列だけ更新
        cols = torch.tensor(self.non_foot_indices, device=self.device, dtype=torch.long)
        self.link_friction[env_ids.unsqueeze(1), cols.unsqueeze(0)] = rand


    def _randomize_foot_link_friction(self, env_ids):
        """足リンクにだけ摩擦をランダム適用（観測用の foot_friction も更新）"""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0 or len(self.feet_indices) == 0:
            return

        mu_min, mu_max = self.env_cfg['foot_friction_range']
        F = len(self.feet_indices)

        rand = mu_min + (mu_max - mu_min) * torch.rand(
            (len(env_ids), F), device=self.device, dtype=gs.tc_float
        )

        self.robot.set_friction_ratio(
            friction_ratio=rand,
            links_idx_local=np.asarray(self.feet_indices, dtype=np.int64),
            envs_idx=env_ids,
        )

        # 全リンクバッファの該当列を更新
        cols = torch.tensor(self.feet_indices, device=self.device, dtype=torch.long)
        self.link_friction[env_ids.unsqueeze(1), cols.unsqueeze(0)] = rand

        # 観測に使う足専用バッファも同じ順序で更新
        self.foot_friction[env_ids] = rand


    def _refresh_total_mass(self, env_ids=None):
        """エンジンから現在のリンク質量を読めればそれを使い、
        だめなら nominal + mass_shift の合計で近似する。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # 1) 物理エンジンから直接読めるなら採用（形状は実装依存）
        try:
            rigid = self.scene.sim.rigid_solver
            mm = torch.as_tensor(rigid.links_info.inertial_mass, device=self.device, dtype=gs.tc_float)
            if mm.ndim == 2:              # (N, L)
                self.total_mass_buf[env_ids] = mm[env_ids].sum(dim=1)
                return
            elif mm.ndim == 1:            # (L,) 共有質量（全env同一）
                self.total_mass_buf[env_ids] = mm.sum()
                return
        except Exception:
            pass

        # 2) フォールバック：名目＋シフトの和
        shift_sum = self.mass_shift_buf[env_ids].sum(dim=1)  # (|env_ids|,)
        self.total_mass_buf[env_ids] = self.nominal_total_mass + shift_sum

    def _randomize_base_mass(self, env_ids):
        min_mass, max_mass = self.env_cfg['added_mass_range']
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        mass_shift = min_mass + (max_mass - min_mass) * torch.rand(
            len(env_ids), len(self.base_link_index), device=self.device, dtype=gs.tc_float
        )
        ls_idx_local = torch.as_tensor(self.base_link_index, device=self.device, dtype=torch.int32)

        # 物理エンジンへ適用
        self.robot.set_mass_shift(
            mass_shift=mass_shift,
            links_idx_local=ls_idx_local,
            envs_idx=env_ids
        )

        # ❶ 自前の mass_shift_buf にも反映
        # （base_link 以外も今後いじるなら同様に列を更新）
        self.mass_shift_buf[env_ids[:, None], ls_idx_local[None, :]] = mass_shift

        # ❷ 総質量を更新
        self._refresh_total_mass(env_ids=env_ids)


    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.env_cfg['com_displacement_range']
        com_shift = min_displacement + (max_displacement - min_displacement) * torch.rand(len(env_ids), len(self.base_link_index), 3, device=self.device)
        ls_idx_local = torch.tensor(self.base_link_index, device=self.device, dtype=torch.int32)


        # Apply only to base_link in the selected environments
        self.robot.set_COM_shift(
            com_shift=com_shift,
            links_idx_local=ls_idx_local,  # Wrap it in a list
            envs_idx=env_ids  # Apply only to specific environments
        )


    def _randomize_motor_strength(self, env_ids):

        min_strength, max_strength = self.env_cfg['motor_strength_range']
        self.motor_strengths[env_ids, :] = gs.rand((len(env_ids), 1), dtype=float) \
                                           * (max_strength - min_strength) + min_strength

    def _randomize_motor_offset(self, env_ids):

        min_offset, max_offset = self.env_cfg['motor_offset_range']
        self.motor_offsets[env_ids, :] = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                                         * (max_offset - min_offset) + min_offset

    def _randomize_kp(self, env_ids):

        min_scale, max_scale = self.env_cfg['kp_scale_range']
        kp_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                    * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):

        min_scale, max_scale = self.env_cfg['kd_scale_range']
        kd_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                    * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]


    def _draw_debug_vis(self):
        self.scene.clear_debug_objects()

        VEL_LENGTH_SCALE = 0.3
        ANGVEL_SCALE     = 0.4
        VEL_RADIUS       = 0.05

        # Draw for every environment that is being shown in the viewer
        for env_idx in self.rendered_envs_idx:
            # ─── origin slightly above the robot base ─────────────────────
            origin = self.base_pos[env_idx].clone().cpu()
            origin[2] += 0.2

            # ─── BLUE arrow: current base-frame linear velocity ───────────
            vel_body  = self.base_lin_vel[env_idx].unsqueeze(0)  # (1,3)
            vel_world = transform_by_quat(
                vel_body,
                self.base_quat[env_idx].unsqueeze(0)
            )[0].cpu()
            self.scene.draw_debug_arrow(
                pos=origin,
                vec=vel_world * VEL_LENGTH_SCALE,
                radius=VEL_RADIUS,
                color=(0.0, 0.0, 1.0, 0.8)
            )

            # ─── GREEN arrow: commanded velocity (rotated to world frame) ─
            # Avoid torch.tensor(list_of_tensors) to prevent copy-construct warning
            cmd_body = torch.cat(
                (self.commands[env_idx, :2], torch.zeros(1, device=self.device, dtype=gs.tc_float)),
                dim=0,
            ).unsqueeze(0)
            cmd_world = transform_by_quat(
                cmd_body,
                self.base_quat[env_idx].unsqueeze(0)
            )[0].cpu()
            self.scene.draw_debug_arrow(
                pos=origin,
                vec=cmd_world * VEL_LENGTH_SCALE,
                radius=VEL_RADIUS,
                color=(0.0, 1.0, 0.0, 0.8)
            )

            # ─── RED arrow: commanded angular velocity (yaw rate command) ─────
            ang_vel_cmd_z = self.commands[env_idx, 2].item()  # commanded yaw rate (ω_cmd)

            # Only draw if nonzero
            if abs(ang_vel_cmd_z) > 1e-5:
                # Use local Y-axis as visual direction, scaled by ω_cmd
                yaw_dir_body = torch.tensor(
                    [0.0, ang_vel_cmd_z, 0.0],
                    device=self.device, dtype=gs.tc_float
                ).unsqueeze(0)

                yaw_dir_world = transform_by_quat(
                    yaw_dir_body,
                    self.base_quat[env_idx].unsqueeze(0)
                )[0].cpu()

                self.scene.draw_debug_arrow(
                    pos=origin,
                    vec=yaw_dir_world * ANGVEL_SCALE,
                    radius=VEL_RADIUS,
                    color=(1.0, 0.0, 0.0, 0.8)
                )

            # --- Goal marker & direction (if goal is assigned) -----------------
            if hasattr(self, "has_goal") and hasattr(self, "goal_pos") and self.has_goal[env_idx]:
                goal = self.goal_pos[env_idx].detach().cpu()
                # Marker at goal location
                self.scene.draw_debug_spheres(
                    poss=goal.unsqueeze(0),
                    radius=0.06,
                    color=(1.0, 0.6, 0.1, 0.9),  # amber
                )
                # Arrow from base toward goal (slightly above base to avoid ground)
                base_pos = self.base_pos[env_idx].detach().cpu()
                vec_to_goal = goal - base_pos
                if torch.norm(vec_to_goal) > 1e-4:
                    lift = torch.tensor(
                        [0.0, 0.0, 0.15],
                        device=base_pos.device,
                        dtype=base_pos.dtype,
                    )
                    self.scene.draw_debug_arrow(
                        pos=base_pos + lift,
                        vec=vec_to_goal,
                        radius=0.03,
                        color=(1.0, 0.4, 0.0, 0.8),  # darker orange
                    )

        # Visualize height patch for the first rendered env (like a LiDAR grid) if enabled
        if self.height_patch_enabled and len(self.rendered_envs_idx) > 0:
            env_idx = self.rendered_envs_idx[0]
            base_xy = self.base_pos[env_idx, :2]
            z_grid = self.height_patch_world[env_idx]  # world heights (G,G)
            x_grid = base_xy[0] + self.height_patch_offsets[..., 0]
            y_grid = base_xy[1] + self.height_patch_offsets[..., 1]
            points = torch.stack((x_grid, y_grid, z_grid), dim=-1).reshape(-1, 3).cpu()
            self.scene.draw_debug_spheres(
                poss=points,
                radius=0.015,
                color=(0.2, 0.8, 1.0, 0.7),  # cyan-ish
            )



    # ------------ reward functions----------------

    # def _reward_tracking_lin_vel(self):
    #     # 誤差（二乗和）
    #     cmd_xy = self.commands[:, :2]
    #     vel_xy = self.base_lin_vel[:, :2]
    #     lin_vel_error = torch.sum((cmd_xy - vel_xy)**2, dim=1)

    #     # スケール（従来どおり）
    #     cmd_vel_norm = torch.norm(cmd_xy, dim=1)
    #     sigma = torch.clamp(
    #         cmd_vel_norm,
    #         min=self.reward_cfg["tracking_min_sigma"],
    #         max=self.reward_cfg["tracking_max_sigma"],
    #     )
    #     reward_full = torch.exp(-lin_vel_error / (sigma + 1e-8))

    #     # コマンドがゼロなら評価しない（寄与0）
    #     eps = 1e-6 #$self.reward_cfg.get("lin_vel_tracking_eps", 1e-6)
    #     active = (cmd_vel_norm > eps).float()
    #     # return reward_full * active
    #     return reward_full

    # def _reward_untracking_lin_vel(self):
    #     """
    #     目標線形速度 (x,y) からズレるほど大きくなるペナルティ
    #     （0=完全一致, 1≒大ズレ）

    #     ガウス形：
    #         pen = 1 - exp(-alpha * e^2 / sigma)

    #     備考:
    #     - sigma: コマンド速度の大きさでスケーリング
    #             （低速時は厳しめ、高速時は緩め）
    #     - alpha: 鋭さを調整する係数（reward_cfg["tracking_lin_alpha"]）
    #     """
    #     # 誤差（二乗和）
    #     e2 = torch.sum(
    #         torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
    #         dim=1
    #     )

    #     # スケール（コマンド速度の大きさ、極小はminで保護）
    #     sigma = torch.clamp(
    #         torch.norm(self.commands[:, :2], dim=1),
    #         min=self.reward_cfg["tracking_min_sigma"],
    #         max=self.reward_cfg["tracking_max_sigma"]
    #     )

    #     # 近傍の鋭さ（未設定なら1.0）
    #     alpha = self.reward_cfg.get("tracking_lin_alpha", 1.0)

    #     # 0(良)→1(悪) に正規化されたガウスペナルティ
    #     return 1.0 - torch.exp(-1.0 * e2 / (sigma + 1e-8))



    # def _reward_tracking_lin_vel_x(self):
    #     """
    #     直進方向(x)の速度追従ごほうび。
    #     近いほど1、外れるほど0へ（ガウス形）。
    #     """
    #     # 誤差（二乗）
    #     e2 = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])

    #     # スケール（コマンドxの絶対値でスケーリング）
    #     sigma = torch.clamp(
    #         torch.abs(self.commands[:, 0]),
    #         min=self.reward_cfg["tracking_min_sigma"],
    #         max=self.reward_cfg["tracking_max_sigma"]
    #     )

    #     # 鋭さ
    #     alpha = self.reward_cfg.get("tracking_lin_alpha_x",
    #             self.reward_cfg.get("tracking_lin_alpha", 1.0))

    #     # デッドバンド（任意）：ほぼ0指令時は評価しない
    #     deadband = self.reward_cfg.get("lin_cmd_deadband_x", 0.0)
    #     mask = (torch.abs(self.commands[:, 0]) > deadband).float()

    #     return torch.exp(-alpha * e2 / (sigma + 1e-8)) * mask + (1.0 - mask)  # ゲート外は中立=1

    def _reward_untracking_lin_vel_x(self):
        """
        直進方向(x)の非追従ペナルティ。
        一致で0、大ズレで1へ（ガウス形ペナルティ）。
        """
        e2 = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])

        sigma = torch.clamp(
            torch.abs(self.commands[:, 0]),
            min=self.reward_cfg["tracking_min_sigma"],
            max=self.reward_cfg["tracking_max_sigma"]
        )

        alpha = self.reward_cfg.get("tracking_lin_alpha_x",
                self.reward_cfg.get("tracking_lin_alpha", 1.0))

        deadband = self.reward_cfg.get("lin_cmd_deadband_x", 0.0)
        mask = (torch.abs(self.commands[:, 0]) > deadband).float()

        # 0(良)→1(悪) に正規化されたガウスペナルティ
        pen = 1.0 - torch.exp(-alpha * e2 / (sigma + 1e-8))
        return pen * mask  # ゲート外は0（評価しない）




    # def _reward_tracking_ang_vel(self):
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     cmd_sigma = torch.clamp(torch.abs(self.commands[:, 2]),
    #                             min=self.reward_cfg["tracking_min_sigma"],
    #                             max=self.reward_cfg["tracking_max_sigma"])
    #     return torch.exp(-ang_vel_error / cmd_sigma)


    def _reward_untracking_ang_vel(self):
        """
        目標角速度からズレるほど大きくなるペナルティ（0=完全一致, 1≒大ズレ）
        ガウス形：pen = 1 - exp(-alpha * e^2 / sigma)

        備考:
        - sigma はコマンド大きさでスケール（低速時は厳しめ / 高速時はゆるめ）
        - alpha で“鋭さ”を調整（大きくすると近傍だけ強く評価）
        """
        # 誤差（二乗）
        e2 = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])

        # スケール（コマンドの大きさでスケール、極小はminで保護）
        sigma = torch.clamp(torch.abs(self.commands[:, 2]),
                            min=self.reward_cfg["tracking_min_sigma"],
                            max=self.reward_cfg["tracking_max_sigma"])

        # 近傍の鋭さ（無ければ1.0）
        alpha = self.reward_cfg.get("tracking_ang_alpha", 1.0)

        # 0(良)→1(悪) に正規化されたガウスペナルティ
        return 1.0 - torch.exp(-alpha * e2 / (sigma + 1e-8))


    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     # return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    #     cmd_vel_norm = torch.clamp(torch.norm(self.commands[:, :2], dim=1), min=self.reward_cfg["tracking_min_sigma"], max=self.reward_cfg["tracking_max_sigma"])
    #     return torch.exp(-lin_vel_error / cmd_vel_norm)

    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     # return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])
    #     cmd_vel_norm = torch.clamp(torch.square(self.commands[:, 2]), min=self.reward_cfg["tracking_min_sigma"], max=self.reward_cfg["tracking_max_sigma"])
    #     return torch.exp(-ang_vel_error / cmd_vel_norm)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_max_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_max_sigma"])

    def _reward_roll_penalty(self):
        # Penalize large roll (base_euler[:, 0] is roll in radians)
        return torch.square(self.base_euler[:, 0])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_default_pose_when_idle(self):
        # Detect idle condition (no command given)
        no_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)

        # Squared distance from default pose (mean to normalize across joints)
        deviation = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

        # Only active when idle; sign and scaling applied outside
        return deviation * no_cmd.float()


    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_relative_base_height(self):
        # --- 現在の相対高さ --------------------------------------------------
        terrain_h = self.get_terrain_height_at_for_base(self.base_pos[:, 0], self.base_pos[:, 1])
        rel_h     = self.base_pos[:, 2] - terrain_h          # shape = (N,)

        # --- 目標との差分 (低すぎるときだけ) ---------------------------------
        target  = self.reward_cfg["relative_base_height_target"]
        penalty = torch.square(torch.relu(target - rel_h))   # shape = (N,)

        # --- ピッチ角が ±10° 以内の環境だけ有効にする -------------------------
        # base_euler[:, 1] は pitch (deg) で格納されている前提
        pitch_ok = (torch.abs(self.base_euler[:, 1]) < 5.0) # Bool tensor shape = (N,)
        penalty  = penalty * pitch_ok.float()                # True →そのまま, False→0
        return penalty

    def _reward_collision(self):
        """
        Penalize collisions on selected bodies.
        Returns the per-env penalty value as a 1D tensor of shape (n_envs,).
        """
        undesired_forces = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        collisions = (undesired_forces > 0.1).float()  # shape (n_envs, len(...))
        
        # Sum over those links to get # of collisions per environment
        return collisions.sum(dim=1)

    def _reward_head_collision(self):
        """
        Penalize collisions on selected bodies.
        Returns the per-env penalty value as a 1D tensor of shape (n_envs,).
        """
        undesired_forces = torch.norm(self.contact_forces[:, self.head_indices, :], dim=-1)
        collisions = (undesired_forces > 0.1).float()  # shape (n_envs, len(...))
        
        # Sum over those links to get # of collisions per environment
        return collisions.sum(dim=1)

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Iterate over legs (order: FL, FR, RL, RR)
        for i in range(self.feet_num):
            # Determine if the current phase indicates a stance phase (< 0.55)
            is_stance = self.leg_phase[:, i] < 0.55

            # Check if the foot is in contact with the ground
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1

            # Reward correct contact behavior (stance matches contact)
            res += ~(contact ^ is_stance)  # XOR for mismatch, negate for correct match

        return res

    def _reward_front_thigh_forward_limit(self):
        """
        Penalise FL and FR thigh joints if they swing too far forward.
        • Angles < THRESH (default −0.4 rad) are considered “excess”.
        • Penalty = (excess)^2  summed over the two joints.
        """
        THRESH = -0.3                                               # radians

        # ‣ self.thigh_dof_pos   shape: (N_envs, 4)  order = FR, FL, RL, RR
        front_angles = self.thigh_dof_pos[:, :2]                    # FR & FL

        # positive value only when angle < THRESH
        excess = torch.relu(THRESH - front_angles)                  # (N, 2)
        penalty = torch.sum(excess**2, dim=1)                       # (N,)

        return penalty                                              # higher ⇒ worse

    def _reward_hip_pos(self):
        diff = self.hip_pos - self.default_hip_pos
        return torch.sum(diff**2, dim=1)

    def _reward_front_hip(self):
        diff = self.hip_pos - self.default_hip_pos
        return torch.sum(diff[:, self.front_hip_idx]**2, dim=1)

    def _reward_rear_hip(self):
        diff = self.hip_pos - self.default_hip_pos
        return torch.sum(diff[:, self.rear_hip_idx]**2, dim=1)


    def _reward_balance(self):
        feet_forces, contact_mask, contact_mask_f = self._get_feet_forces_and_contact()

        # スイング中のノイズ低減（接触中のみ評価）
        feet_forces = feet_forces * contact_mask_f.unsqueeze(-1)   # (N,4,3)

        # 左右（例：0,2） vs （1,3） ※足の順番に合わせて調整
        group_a = (0, 2)
        group_b = (1, 3)
        F_a = feet_forces[:, group_a, :].sum(dim=1)   # (N,3)
        F_b = feet_forces[:, group_b, :].sum(dim=1)   # (N,3)
        diff = F_a - F_b                              # (N,3)

        penalty = torch.linalg.norm(diff, dim=1)      # (N,)

        # mg で正規化（ランダム化反映済みの総質量を使用）
        if not hasattr(self, "total_mass_buf") or (self.total_mass_buf == 0).any():
            self._refresh_total_mass()

        mg = self.total_mass_buf * self.g_mag         # (N,)
        penalty = penalty / (mg + 1e-6)

        return penalty  # ※報酬スケール側で負にする運用ならこのままでOK


    def _reward_front_feet_swing_height(self):
        # 地面との接触判定（front feet）
        contact = torch.norm(self.contact_forces[:, self.feet_front_indices, :3], dim=2) > 1.0
        
        # 前足の位置 (z成分)
        z = self.feet_front_pos[:, :, 2]

        # 地形の高さ
        terrain_h = self.get_terrain_height_at(
            self.feet_front_pos[:, :, 0],
            self.feet_front_pos[:, :, 1]
        )

        # 高さ差（地面との相対高度）
        rel_h = z - terrain_h  # [num_envs, 2]

        # 前足の速度（x方向）を feet_vel から抽出
        front_feet_vel = self.feet_vel[:, :2, :]          # shape (N, 2, 3)
        foot_vel_x     = front_feet_vel[:, :, 0]          # x‑velocity
        # 条件：接地していないかつ前進中
        swing_forward = (~contact) & (foot_vel_x > 0.05)

        # 高さが足りない場合にペナルティ
        height_error = torch.relu(self.step_height_for_front - rel_h)
        pos_error = height_error * swing_forward
        # 報酬（低すぎるスイング足をペナルティ）
        reward = torch.sum(pos_error, dim=1)
        return reward

    def _reward_front_feet_clearance(self):
        """
        Encourage front feet to lift off the terrain (swing),
        but only when the robot is tracking the commanded velocity (x, y).

        - Reward grows quadratically with height above 1 cm.
        - Gated by 2D velocity tracking error.
        """
        contact = torch.norm(
            self.contact_forces[:, self.feet_front_indices, :3], dim=2
        ) > 1.0  # (N, 2)

        z = self.feet_front_pos[:, :, 2]
        terrain_h = self.get_terrain_height_at(
            self.feet_front_pos[:, :, 0],
            self.feet_front_pos[:, :, 1]
        )
        rel_h = z - terrain_h  # (N, 2)

        swing_motion = ~contact

        clearance_start = 0.05
        max_reward_height = 0.15
        max_bonus = (max_reward_height - clearance_start) ** 2

        above_target = torch.clamp(rel_h - clearance_start, min=0.0)
        bonus = above_target.pow(2).clamp(max=max_bonus)
        bonus = bonus * swing_motion

        # 2D velocity tracking gate (x and y)
        vel_xy = self.base_lin_vel[:, :2]   # (N, 2)
        cmd_xy = self.commands[:, :2]       # (N, 2)
        vel_error_norm_sq = torch.sum((vel_xy - cmd_xy) ** 2, dim=1)  # (N,)
        vel_tracking_weight = torch.exp(-vel_error_norm_sq / 0.04)    # (N,)
        bonus = bonus * vel_tracking_weight.unsqueeze(1)

        reward = torch.sum(bonus, dim=1)  # (N,)
        return reward

    def _reward_rear_feet_clearance(self):
        """
        Encourage rear feet to lift off the terrain (swing),
        but only when the robot is tracking the commanded velocity (x, y).

        - Reward grows quadratically with height above 1 cm.
        - Capped at 10 cm.
        - Gated by 2D velocity tracking error.
        """
        # 1) Check rear foot contact
        contact = torch.norm(
            self.contact_forces[:, self.feet_rear_indices, :3], dim=2
        ) > 1.0  # (N, 2)

        # 2) Relative foot height above terrain
        z = self.feet_rear_pos[:, :, 2]
        terrain_h = self.get_terrain_height_at(
            self.feet_rear_pos[:, :, 0],
            self.feet_rear_pos[:, :, 1]
        )
        rel_h = z - terrain_h  # (N, 2)

        # 3) Only reward if feet are off the ground
        swing_motion = ~contact  # (N, 2)

        # 4) Quadratic height reward
        clearance_start = 0.05
        max_reward_height = 0.15
        max_bonus = (max_reward_height - clearance_start) ** 2

        above_target = torch.clamp(rel_h - clearance_start, min=0.0)
        bonus = above_target.pow(2).clamp(max=max_bonus)
        bonus = bonus * swing_motion

        # 5) 2D velocity tracking gate (x and y)
        vel_xy = self.base_lin_vel[:, :2]   # (N, 2)
        cmd_xy = self.commands[:, :2]       # (N, 2)
        vel_error_norm_sq = torch.sum((vel_xy - cmd_xy) ** 2, dim=1)  # (N,)
        vel_tracking_weight = torch.exp(-vel_error_norm_sq / 0.04)    # (N,)
        bonus = bonus * vel_tracking_weight.unsqueeze(1)  # (N, 2)

        # 6) Total reward per environment
        reward = torch.sum(bonus, dim=1)  # (N,)
        return reward

    def _reward_calf_clearance(self):
        """
        Foot-style clearance reward for all calves (4×):
            • target height is measured in the body frame
            • reward = (height-error)^2  ×  lateral speed of the calf tip
        """
        # --- World-frame position & velocity of each calf ----------------
        calf_world_pos = self.calf_pos                # (N, 4, 3)
        calf_world_vel = self.calf_vel                # (N, 4, 3)

        # ---  Convert to body frame --------------------------------------
        pos_body = self._world_to_base_transform(
            calf_world_pos, self.base_pos, self.base_quat)      # (N,4,3)

        vel_body = self._world_to_base_transform(
            calf_world_vel, torch.zeros_like(self.base_pos), self.base_quat)

        # --- Height error (z) -------------------------------------------
        target_h = self.reward_cfg.get("calf_clearance_height_target")
        height_err = torch.square(pos_body[:, :, 2] - target_h)           # (N,4)

        # --- Lateral velocity magnitude (x-y plane) ----------------------
        lateral_vel = torch.norm(vel_body[:, :, :2], dim=2)               # (N,4)

        # --- Final reward ------------------------------------------------
        clearance_reward = height_err * lateral_vel                       # (N,4)
        return clearance_reward.sum(dim=1)                                # (N,)

    def _reward_rear_feet_swing_height(self):
        # Determine which rear feet are in contact
        contact = torch.norm(self.contact_forces[:, self.feet_rear_indices, :3], dim=2) > 1.0
    
        # Extract x, y, z from rear foot positions (shape: [num_envs, 2])
        x = self.feet_rear_pos[:, :, 0]
        y = self.feet_rear_pos[:, :, 1]
        z = self.feet_rear_pos[:, :, 2]
    
        # Terrain height under each foot (shape: [num_envs, 2])
        terrain_h = self.get_terrain_height_at(x, y)
    
        # Relative height above terrain
        rel_h = z - terrain_h
    
        # Penalize only if the swing foot is *below* the desired height
        height_error = torch.relu(self.step_height_for_rear - rel_h)  # only penalize if below
        pos_error = height_error * ~contact  # only apply to swing feet
    
        reward = torch.sum(pos_error, dim=1)  # sum over the two rear feet
        return reward

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)


    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits).clip(min=0.), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        # return 1.0
        return self.reset_buf & ~self.time_out_buf & ~self.out_of_bounds_buf

    # def _reward_fixed_termination(self):
    #     return -1.0

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.reward_cfg["max_contact_force"]).clip(min=0.), dim=1)

    def _reward_feet_stumble(self):
        # Detect which feet are in contact (force > threshold)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=2) > 1.0  # (batch, n_feet)

        # Horizontal velocities of feet (ignore vertical z component)
        feet_xy_vel = self.feet_vel[:, :, :2]  # (batch, n_feet, 2)

        # Magnitude of horizontal velocity
        slip_mag = torch.norm(feet_xy_vel, dim=2)  # (batch, n_feet)

        # Only count feet that are in contact
        slip_when_contact = slip_mag * contact.float()

        # Aggregate over feet (mean = normalized, sum = stronger penalty)
        return torch.mean(slip_when_contact, dim=1)

    def _reward_base_upward_progress(self):
        """ 階段登攀時のベース上昇に対する報酬（連続的に上に移動する場合に与える） """
        terrain_h = self.get_terrain_height_at_for_base(self.base_pos[:, 0], self.base_pos[:, 1])
        rel_h = self.base_pos[:, 2] - terrain_h

        # 差分で高さの変化を取る（上昇時のみ報酬）
        delta_h = rel_h - getattr(self, "prev_rel_h", torch.zeros_like(rel_h))
        self.prev_rel_h = rel_h.clone()

        return torch.relu(delta_h)  # 上昇した分だけ報酬


    def _reward_rear_feet_level_with_front(self):
        # Get world-frame foot positions
        front_z = self.feet_front_pos[:, :, 2]  # (N, 2)
        rear_z  = self.feet_rear_pos[:, :, 2]   # (N, 2)

        # Terrain height under each foot
        terrain_front = self.get_terrain_height_at(
            self.feet_front_pos[:, :, 0], self.feet_front_pos[:, :, 1])
        terrain_rear = self.get_terrain_height_at(
            self.feet_rear_pos[:, :, 0], self.feet_rear_pos[:, :, 1])

        # Relative clearance above terrain
        front_rel = front_z - terrain_front  # (N, 2)
        rear_rel  = rear_z  - terrain_rear   # (N, 2)

        # Diagonal pairs: [FL, FR], [RL, RR]
        # Assume index 0 = FL, 1 = FR, 0 = RL, 1 = RR
        # Compare: FL vs RR, FR vs RL
        rel_FR = front_rel[:, 0]
        rel_FL = front_rel[:, 1]
        rel_RR = rear_rel[:, 0]
        rel_RL = rear_rel[:, 1]

        # Contact info: feet in contact = True
        contact_front = torch.norm(self.contact_forces[:, self.feet_front_indices, :3], dim=2) > 1.0  # (N, 2)
        contact_rear  = torch.norm(self.contact_forces[:, self.feet_rear_indices,  :3], dim=2) > 1.0  # (N, 2)

        # Diagonal contact status
        FR_contact = contact_front[:, 0]
        FL_contact = contact_front[:, 1]
        RR_contact = contact_rear[:, 0]
        RL_contact = contact_rear[:, 1]

        # diag1 = FL–RR; reward if at least one of them is in the air
        diag1_mask = ~(FL_contact & RR_contact)  # reward if not both in contact
        diag1_diff = torch.abs(rel_FL - rel_RR)
        reward_diag1 = torch.exp(-10.0 * diag1_diff) * diag1_mask.float()

        # diag2 = FR–RL; reward if at least one of them is in the air
        diag2_mask = ~(FR_contact & RL_contact)
        diag2_diff = torch.abs(rel_FR - rel_RL)
        reward_diag2 = torch.exp(-10.0 * diag2_diff) * diag2_mask.float()

        # Total reward
        return reward_diag1 + reward_diag2


    def _reward_calf_collision_low_clearance(self):
        penalty = torch.zeros(self.num_envs, device=self.device)

        for calf_idx, foot_idx in self.calf_to_foot_map.items():
            # 1. Check if calf is in contact
            calf_force = torch.norm(self.contact_forces[:, calf_idx, :], dim=1)
            calf_collision = calf_force > 1.0  # shape: (N,)
            feet_tensor_idx = self.link_idx_to_feet_tensor_idx.get(foot_idx, None)
            if feet_tensor_idx is None:
                continue  # skip if not found

            foot_z = self.feet_pos[:, feet_tensor_idx, 2]
            terrain_z = self.get_terrain_height_at(
                self.feet_pos[:, feet_tensor_idx, 0],
                self.feet_pos[:, feet_tensor_idx, 1]
            )
            
            clearance = foot_z - terrain_z

            # 3. Penalize if clearance is low while calf is in contact
            low_clearance = clearance < 0.05
            penalty += calf_collision.float() * low_clearance.float()

        return penalty


    def _reward_feet_distance_diff(self):
        """
        Penalize front/rear feet if their x-position in base frame moves too far
        from the default position. Allow some margin (e.g., 10cm) before penalizing.
        The penalty grows exponentially with the exceeded amount.
        """
        margin = 0.2  # [m] 許容距離
        scale = 10.0   # 指数スケーリング係数

        # デフォルト位置（ベースフレーム基準）
        default_front_x = 0.1934  #self.env_cfg["default_feet_pos_base"]["front"]  # 例: 0.25
        default_rear_x  = -0.1934 #self.env_cfg["default_feet_pos_base"]["rear"]   # 例: -0.25

        # 現在の足位置（ベースフレームでの x 座標）
        front_x = self.front_feet_pos_base[:, :, 0]  # shape: (N, 2)
        rear_x  = self.rear_feet_pos_base[:, :, 0]   # shape: (N, 2)

        # ⛔ 差分そのものは「絶対値」で固定（変更しない）
        front_diff = torch.abs(front_x - default_front_x)  # (N, 2)
        rear_diff  = torch.abs(rear_x  - default_rear_x)   # (N, 2)

        # ✅ マージン以下ならゼロ、それを超えた分だけ指数罰
        front_excess = torch.relu(front_diff - margin)  # (N, 2)
        rear_excess  = torch.relu(rear_diff  - margin)  # (N, 2)

        # 📈 指数ペナルティ（調整可能）
        penalty_front = torch.exp(scale * front_excess) - 1.0
        penalty_rear  = torch.exp(scale * rear_excess)  - 1.0

        # 🎯 合計
        total_penalty = penalty_front.sum(dim=1) + penalty_rear.sum(dim=1)
        return total_penalty


    def _reward_front_leg_retraction(self):
        # FLとFRのx方向ベースフレーム位置（大きすぎると前に伸びすぎ）
        front_x = self.front_feet_pos_base[:, :, 0]
        penalty = torch.relu(front_x - 0.4)  # デフォルト位置より前に出すぎたら罰
        return penalty.sum(dim=1)

    def _reward_thigh_retraction(self):
        return -self.thigh_pos_base[:, :, 0].mean(dim=1)



    def _reward_foot_clearance(self):
        # 1. 足のワールド位置・速度を取得
        foot_world_pos = self.feet_pos             # shape: (N, 4, 3)
        foot_world_vel = self.feet_vel             # shape: (N, 4, 3)

        # 2. ボディ中心位置と速度を減算（相対座標系に変換）
        rel_pos = foot_world_pos - self.base_pos.unsqueeze(1)     # shape: (N, 4, 3)
        rel_vel = foot_world_vel - self.base_lin_vel.unsqueeze(1) # shape: (N, 4, 3)

        # 3. ボディ座標系に変換（回転のみ適用）
        pos_body = self._world_to_base_transform(foot_world_pos, self.base_pos, self.base_quat)  # shape: (N, 4, 3)
        vel_body = self._world_to_base_transform(foot_world_vel, torch.zeros_like(self.base_pos), self.base_quat)

        # 4. z方向の高さ誤差（二乗誤差）
        target_height = self.reward_cfg["foot_clearance_height_target"]  # e.g. 0.1 [m]
        height_error = torch.square(pos_body[:, :, 2] - target_height)  # shape: (N, 4)

        # 5. 横方向の速度（x-y平面のnorm）
        lateral_vel = torch.norm(vel_body[:, :, :2], dim=2)  # shape: (N, 4)

        # 6. 報酬計算（高さ誤差 × 横速度）
        clearance_reward = height_error * lateral_vel

        return torch.sum(clearance_reward, dim=1)  # shape: (N,)


    def _reward_powers(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)

    def _reward_stand_still(self):
        # Only penalize motion when there is truly no motion command
        no_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        # Penalize actual base motion (more aligned with the intent than joint pose drift)
        motion = torch.norm(self.base_lin_vel[:, :2], dim=1) + torch.abs(self.base_ang_vel[:, 2])
        return motion * no_cmd.float()

    def _reward_idle_leg_raise(self):
        """
        Penalize keeping any leg raised for longer than 1 second while idle.
        The timer resets when the leg touches down or a motion command is issued.
        """
        no_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        idle_mask = no_cmd.unsqueeze(1)  # (N,1) for broadcasting

        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0  # (N, feet)
        legs_raised = (~contact) & idle_mask

        # Accumulate raised duration only when idle; reset otherwise.
        self.idle_leg_raise_duration = torch.where(
            legs_raised,
            self.idle_leg_raise_duration + self.dt,
            torch.zeros_like(self.idle_leg_raise_duration),
        )

        # Penalty starts after 1 second of continuous raised state.
        over_duration = torch.clamp(self.idle_leg_raise_duration - 1.0, min=0.0)
        penalty = over_duration.sum(dim=1)
        return penalty * no_cmd.float()

    def _reward_all_feet_contact_when_idle(self):
        # Reward having all feet in contact when no motion is commanded and the robot is upright
        no_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        upright = (torch.abs(self.base_euler[:, 0]) < math.radians(30.0)) & (
            torch.abs(self.base_euler[:, 1]) < math.radians(30.0)
        )
        active_mask = no_cmd & upright
        if not torch.any(active_mask):
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0  # (N, feet)
        all_feet_contact = contact.all(dim=1).float()
        return all_feet_contact * active_mask.float()


    def _reward_both_front_feet_airborne(self):
        """
        Penalize when both front feet are off the ground simultaneously.
        Encourages at least one front foot to maintain contact for stability.

        Returns:
            penalty: shape (N,), 1.0 if both are off the ground, 0.0 otherwise
        """
        # 1. 前足（左右）のコンタクト状態を取得
        contact = torch.norm(
            self.contact_forces[:, self.feet_front_indices, :3], dim=2
        ) > 1.0  # shape: (N, 2)

        # 2. 両方とも非接地の場合のペナルティ
        airborne_both = (~contact).all(dim=1).float()  # shape: (N,), True → 1.0
        return airborne_both

    def _reward_both_rear_feet_airborne(self):
        """
        Penalize when both front feet are off the ground simultaneously.
        Encourages at least one front foot to maintain contact for stability.

        Returns:
            penalty: shape (N,), 1.0 if both are off the ground, 0.0 otherwise
        """
        # 1. 前足（左右）のコンタクト状態を取得
        contact = torch.norm(
            self.contact_forces[:, self.feet_rear_indices, :3], dim=2
        ) > 1.0  # shape: (N, 2)

        # 2. 両方とも非接地の場合のペナルティ
        airborne_both = (~contact).all(dim=1).float()  # shape: (N,), True → 1.0
        return airborne_both


    # -------------------- ② swing 中の “足が進んでいない” ------------------------
    def _reward_swing_stuck(self):
        """
        Swing 中なのに足が“ほぼ動いていない” ──
        • 並進 OR 旋回どちらかの基底速度がしきい値を超える
        • そのとき足水平速度＆角速度が十分小さい ⇒ 引っかかり判定
        """
        swing = self.leg_phase > 0.55                         # (N,4)

        # --- フットの “動いていなさ” 判定 -----------------------------------
        foot_v_xy  = torch.norm(self.feet_vel[:, :, :2], dim=2)   # (N,4)
        foot_w_z   = torch.abs(self.feet_vel[:, :, 2])            # (N,4) ← yaw 方向 vel が無い場合は 0 で OK
        foot_stuck = (foot_v_xy < 0.05) & (foot_w_z < 0.05)      # (N,4)

        # --- ベースが “動こうとしている” 判定 -------------------------------
        lin_xy  = torch.norm(self.base_lin_vel[:, :2], dim=1)     # (N,)
        ang_z   = torch.abs(self.base_ang_vel[:, 2])              # (N,)
        body_move = (lin_xy > 0.10) | (ang_z > 0.10)              # (N,)

        # (N,1) → (N,4) にブロードキャスト
        body_move_exp = body_move.unsqueeze(1).expand_as(foot_stuck)

        # --- stuck 条件 ----------------------------------------------------
        stuck = swing & foot_stuck & body_move_exp               # (N,4)

        # --- 継続時間バッファ ---------------------------------------------
        self._stuck_buf = getattr(
            self, "_stuck_buf", torch.zeros_like(stuck, dtype=torch.float)
        )
        self._stuck_buf[stuck]     += self.dt
        self._stuck_buf[~stuck]     = 0.0

        long_stuck = self._stuck_buf > 0.05    # 50 ms 以上続いたらアウト
        penalty    = long_stuck.float().sum(dim=1)   # (N,) 0‥4
        return penalty

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_foot_xy_compact(self):
        """
        後ろ足だけ対象。ペナルティを強めるための3段ブースト付き：
        (A) 脚数補正、(B) 指数強化、(C) マージン付き外側強化
        """
        feet_xy = self.rear_feet_pos_base[..., :2]  # (N, Fr, 2)

        # ====== hyperparams（お好みで調整）======
        ALPHA_SCALE   = 2.0    # (A) 全体倍率：効きが弱いならここを上げる
        P_POWER       = 4.0    # (B) 2なら従来、4以上で大きくはみ出すほど急増
        SAFE_MARGIN   = 0.01   # (C) 矩形をほんの少し内側に縮める[m]（境界ビクつき対策兼ペナルティ強化）
        W_X, W_Y      = 1.2, 1.0  # x方向をより厳しくしたい場合の重み

        # 安全域を内側へ少し縮める（はみ出し検出を早める）
        L = self.body_half_length - SAFE_MARGIN
        W = self.body_half_width  - SAFE_MARGIN

        dx = (feet_xy[..., 0].abs() - L).clamp(min=0.0)
        dy = (feet_xy[..., 1].abs() - W).clamp(min=0.0)

        # 異方性（前後方向を厳しめ etc.）
        dx = dx * W_X
        dy = dy * W_Y

        # 基本は二乗和（従来）
        t2 = dx * dx + dy * dy  # (N, Fr)

        # 大はみ出しをより重く（p>=3で急増）
        if P_POWER != 2.0:
            # t を SAFE_MARGIN で正規化して、margin超えた部分を指数強化
            t  = torch.sqrt(t2 + 1e-8)
            t2 = (t / max(SAFE_MARGIN, 1e-6)).clamp(min=1.0).pow(P_POWER - 2.0) * t2

        # 脚数補正：全脚→後脚に切り替えたことで落ちたスケールを戻す
        Fr     = feet_xy.shape[1]
        F_all  = getattr(self, "num_feet", Fr)
        penalty = t2.sum(dim=1) * (F_all / Fr) * ALPHA_SCALE  # (N,)

        # 設計が「負で返す」ならこちら
        # return -penalty
        return penalty


    def _compute_leg_clearance_penalty(self, include_rear=True, return_depth=False):
        """
        Compute lateral-clearance penalty for left/right pairs.
        Pair ordering is inferred every step (max y = left, min y = right) so it
        works even if link indices are not sorted.
        """
        pairs = []
        if getattr(self, "front_feet_pos_base", None) is not None:
            pairs.append(self.front_feet_pos_base)
        if include_rear and getattr(self, "rear_feet_pos_base", None) is not None:
            pairs.append(self.rear_feet_pos_base)

        if len(pairs) == 0:
            zeros = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            return (zeros, zeros) if return_depth else zeros

        hip_w   = float(self.reward_cfg.get("hip_width", 0.1))
        margin  = float(self.reward_cfg.get("cross_margin", 0.05))
        power   = int(self.reward_cfg.get("cross_power", 2))
        g_soft  = float(self.reward_cfg.get("cross_soft_gain", 1.0))
        c_hard  = float(self.reward_cfg.get("cross_hard_const", 1.0))
        g_hard  = float(self.reward_cfg.get("cross_hard_gain", 2.0))
        deadband = float(self.reward_cfg.get("side_deadband", 0.0))
        cross_simple = bool(self.reward_cfg.get("cross_simple", False))
        eps = 1e-6

        penalties = []
        depths = []
        for pair in pairs:
            y_vals = pair[..., 1]
            y_left, _ = torch.max(y_vals, dim=1)   # larger y → left
            y_right, _ = torch.min(y_vals, dim=1)  # smaller y → right
            gap = y_left - y_right                 # expected positive when not crossing

            if cross_simple:
                # Simple mode: only enforce margin and side (centerline) separation.
                gap_violation    = F.relu(margin - gap) / max(hip_w, eps)
                center_pen       = (F.relu(margin - y_left) + F.relu(margin + y_right)) / max(hip_w, eps)
                penalty = gap_violation + center_pen
                depth_nd = (-gap / max(hip_w, eps)).clamp(min=0)
            else:
                deficit_m   = F.relu(margin - gap)
                deficit_nd  = deficit_m / max(hip_w, eps)
                soft_pen    = g_soft * (deficit_nd ** power)

                depth_nd    = (-gap / max(hip_w, eps)).clamp(min=0)
                hard_pen    = (gap < 0).float() * (c_hard + g_hard * (depth_nd ** power))

                center_pen  = (F.relu(deadband - y_left) + F.relu(deadband + y_right)) / max(hip_w, eps)
                penalty = soft_pen + hard_pen + center_pen

            penalties.append(penalty)
            depths.append(depth_nd)

        penalties = torch.stack(penalties, dim=1).mean(dim=1)
        max_depth = torch.stack(depths, dim=1).max(dim=1).values

        return (penalties, max_depth) if return_depth else penalties

    def _compute_fore_aft_clearance_penalty(self, return_depth=False):
        """
        Fore-aft clearance penalty so front/rear feet stay on their respective
        sides of the base x-axis and keep a margin to avoid collisions.
        """
        front = getattr(self, "front_feet_pos_base", None)
        rear = getattr(self, "rear_feet_pos_base", None)
        if front is None or rear is None:
            zeros = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            return (zeros, zeros) if return_depth else zeros

        margin = float(self.reward_cfg.get("fore_margin", self.reward_cfg.get("cross_margin", 0.05)))
        power = int(self.reward_cfg.get("fore_power", self.reward_cfg.get("cross_power", 2)))
        g_soft = float(self.reward_cfg.get("fore_soft_gain", self.reward_cfg.get("cross_soft_gain", 1.0)))
        c_hard = float(self.reward_cfg.get("fore_hard_const", self.reward_cfg.get("cross_hard_const", 1.0)))
        g_hard = float(self.reward_cfg.get("fore_hard_gain", self.reward_cfg.get("cross_hard_gain", 2.0)))
        deadband = float(self.reward_cfg.get("fore_deadband", 0.0))
        eps = 1e-6
        half_len = max(float(self.body_half_length), eps)

        front_x = front[:, :, 0]
        rear_x = rear[:, :, 0]
        front_y = front[:, :, 1]
        rear_y = rear[:, :, 1]

        # Determine left/right feet on-the-fly to stay robust to link ordering.
        idx = torch.arange(self.num_envs, device=self.device)
        front_left_idx = torch.argmax(front_y, dim=1)
        front_right_idx = 1 - front_left_idx
        rear_left_idx = torch.argmax(rear_y, dim=1)
        rear_right_idx = 1 - rear_left_idx

        front_left_x = front_x[idx, front_left_idx]
        front_right_x = front_x[idx, front_right_idx]
        rear_left_x = rear_x[idx, rear_left_idx]
        rear_right_x = rear_x[idx, rear_right_idx]

        gaps = torch.stack(
            [front_left_x - rear_left_x, front_right_x - rear_right_x], dim=1
        )  # positive when front is ahead of rear

        deficit_m = F.relu(margin - gaps)
        deficit_nd = deficit_m / half_len
        soft_pen = g_soft * (deficit_nd ** power)

        depth_nd = (-gaps / half_len).clamp(min=0)
        hard_pen = (gaps < 0).float() * (c_hard + g_hard * (depth_nd ** power))

        # Keep each foot on its side of the base x-axis with an optional deadband.
        center_pen_front = F.relu(deadband - torch.stack([front_left_x, front_right_x], dim=1)) / half_len
        center_pen_rear = F.relu(deadband + torch.stack([rear_left_x, rear_right_x], dim=1)) / half_len

        penalty = soft_pen + hard_pen + center_pen_front + center_pen_rear
        max_depth = depth_nd.max(dim=1).values

        penalty = penalty.mean(dim=1)
        return (penalty, max_depth) if return_depth else penalty

    def _reward_leg_cross(self):
        """
        Returns: (N,)  正の値（= ペナルティ量）
        Uses base-frame foot positions and enforces clearance for both front/rear.
        See _compute_leg_clearance_penalty for parameter meanings.
        """
        penalty, depth = self._compute_leg_clearance_penalty(include_rear=True, return_depth=True)
        # Keep the latest depth for diagnostics/termination guards.
        self._last_leg_cross_depth = depth
        return penalty

    def _reward_leg_cross_fore_aft(self):
        """
        Penalizes front/rear feet crossing the base x-axis or encroaching toward
        each other to avoid fore-aft collisions.
        """
        penalty, depth = self._compute_fore_aft_clearance_penalty(return_depth=True)
        self._last_leg_cross_fore_depth = depth
        return penalty

    def _reward_action_curvature(self):
        """
        離散二階差分でアクション曲率を近似してペナルティにする。
        """
        a_t   = self.actions          # (N, A)
        a_tm1 = self.prev_actions
        a_tm2 = self.prev_prev_actions

        a_ddot = a_t - 2.0 * a_tm1 + a_tm2
        denom = (1.0 + a_tm1**2).pow(1.5)
        tiny = 1e-6
        racurv_per_dof = torch.abs(a_ddot) / (denom + tiny)

        racurv = racurv_per_dof.sum(dim=1)  # Σ_i
        return racurv

    def _reward_effort_symmetry(self):
        """
        各関節種別(hip/thigh/calf)ごとに脚方向のトルク EMA の std を取ってペナルティ。
        torque_ema: (N, L, J) L=脚数, J=関節数/脚
        """
        # 脚方向で標準偏差 → (N, J)
        std_per_joint = torch.std(self.torque_ema, dim=1)
        reffort = std_per_joint.sum(dim=1)  # Σ over joint-types

        return reffort

    def _reward_stuck_ema(self):
        """
        leg_phase を使わず、足先速度＋接触＋本体速度から
        スイングスタックを検知し、
        ・即時ペナルティ (stuck_instant)
        ・履歴ベースの EMA ペナルティ (swing_stuck_ema)
        の両方を使って罰する。
        """

        # 1) 足先速度（world frame）
        #    feet_vel: (N, num_feet, 3)
        foot_speed = torch.norm(self.feet_vel, dim=2)  # (N, num_feet)

        swing_speed_thresh = self.reward_cfg.get("swing_speed_thresh", 0.4)
        swing_like = (foot_speed > swing_speed_thresh).float()  # (N, num_feet)

        # 2) 接触（1: 接触、0: 非接触）
        _, _, contact_mask_f = self._get_feet_forces_and_contact(force_thresh=1.0)  # (N, num_feet)

        # 3) 「コマンドに見合って進んでいない」かどうかを動的に判定
        #    robot_speed_xy: 実際の水平速度 [m/s]
        robot_speed_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)  # (N,)

        #    cmd_speed_xy: コマンドの水平速度 [m/s]
        cmd_speed_xy = torch.norm(self.commands[:, :2], dim=1)       # (N,)

        #    コマンド速度の何割を下回ったら「遅い」とみなすか
        speed_scale = self.reward_cfg.get("stuck_speed_scale", 0.8)   # fraction of commanded speed
        speed_min = self.reward_cfg.get("stuck_speed_min_thresh", 0.05)

        # thresh = max(speed_min, cmd_speed_xy * speed_scale)
        speed_thresh = torch.maximum(
            torch.as_tensor(speed_min, device=self.device, dtype=gs.tc_float),
            cmd_speed_xy * speed_scale,
        )  # (N,)

        low_speed = (robot_speed_xy < speed_thresh).float().unsqueeze(1)  # (N,1)

        # 4) スタック瞬間指標
        #    「スイングっぽい」かつ「接触している」かつ「進めていない」
        stuck_raw = swing_like * contact_mask_f * low_speed  # (N, num_feet)

        # 5) 各 env で、どれか 1 足でもスタックしていれば 1 に近づける
        stuck_instant = stuck_raw.max(dim=1).values  # (N,)

        # 6) EMA 更新
        #    上昇時は早く、下降時はゆっくり戻る非対称 EMA にすることで
        #    「引っかかった瞬間はすぐ効き、解除後はゆっくり減衰」させる
        alpha_up   = self.reward_cfg.get("stuck_ema_alpha_up",   0.9)
        alpha_down = self.reward_cfg.get("stuck_ema_alpha_down", 0.99)

        rising = stuck_instant > self.swing_stuck_ema  # True: EMA を上げる方向
        alpha = torch.where(
            rising,
            torch.full_like(stuck_instant, alpha_up),
            torch.full_like(stuck_instant, alpha_down),
        )

        self.swing_stuck_ema = alpha * self.swing_stuck_ema + (1.0 - alpha) * stuck_instant

        # 7) 小さな値には反応しない（margin）
        margin = self.reward_cfg.get("stuck_ema_margin", 0.02)
        ema_term = torch.clamp(self.swing_stuck_ema - margin, min=0.0)  # (N,)

        # 8) 瞬間ペナルティも足す
        #    → stuck になった瞬間からペナルティが効き始める
        w_inst = self.reward_cfg.get("stuck_instant_weight", 0.5)
        instant_term = w_inst * stuck_instant  # (N,)

        penalty = ema_term + instant_term  # (N,)

        # 9) ほぼ静止コマンドのときは罰しない（その場ステイ中の接触は無視）
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)  # (N,)
        active_cmd = cmd_speed > self.reward_cfg.get("stuck_min_command", 0.05)
        penalty[~active_cmd] = 0.0

        return penalty

    def _reward_goal_reached(self):
        rew = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        rew[self.goal_reached_flag] = 1.0  # 1回到達ごとに +1 とか
        # 使い終わったらフラグをクリア
        self.goal_reached_flag[:] = False
        return rew
