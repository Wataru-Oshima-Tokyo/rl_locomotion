import numpy as np
import time
import genesis as gs
import torch

########################## 初期化 ##########################
gs.init(backend=gs.gpu)

########################## シーンの作成 ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## エンティティ ##########################

subterrain_size = 4.0
horizontal_scale = 0.05
vertical_scale = 0.005
cols = 1
rows = 1
n_subterrains=(cols, rows)
terrain_types = ["pyramid_shallow_down_stairs_terrain"] # 
grid = [[None for _ in range(cols)] for _ in range(rows)]
grid[0][0] = terrain_types[0]
# Calculate the total width and height of the terrain
total_width = (cols)* subterrain_size
total_height =(rows)* subterrain_size

# Calculate the center coordinates
center_x = total_width / 2
center_y = total_height / 2

terrain  = gs.morphs.Terrain(
    pos=(-center_x, -center_y ,0),
    subterrain_size=(subterrain_size, subterrain_size),
    n_subterrains=n_subterrains,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    subterrain_types=grid
)
scene.add_entity(terrain)
# go2 = scene.add_entity(
#     gs.morphs.MJCF(
#         file  = 'urdf/luvbit/luvbit_prototype00.urdf',
#     ),
# )
# base_init = [0.0, 0.0, 0.3]
# pos_base_init = torch.tensor(base_init, device="cuda:0")
# base_quat = [1.0, 0.0, 0.0, 0.0]
# base_init_quat = torch.tensor(base_quat,  device="cuda:0")
# scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
# robot= scene.add_entity(
#     gs.morphs.URDF(
#         file='urdf/luvbit/urdf/luvbit_prototype.urdf',
#         # merge_fixed_links=False,
#         pos=pos_base_init.cpu().numpy(),
#         quat=base_init_quat.cpu().numpy()
#     ),
# )

# robot = scene.add_entity(
#     gs.morphs.MJCF(
#         file  = 'xml/luvBit/luvBit.xml',
#     ),
# )

########################## ビルド ##########################
scene.build()

# 1. List your GO2 quadruped joints in the desired order
# jnt_names = [
#     "left_hip_yaw_joint",
#     "left_hip_roll_joint",
#     "left_hip_pitch_joint",
#     "left_knee_joint",
#     "left_ankle_joint",
#     "right_hip_yaw_joint",
#     "right_hip_roll_joint",
#     "right_hip_pitch_joint",
#     "right_knee_joint",
#     "right_ankle_joint",
#     "neck_pitch_joint",
#     "neck_yaw_joint",
# ]

# 2. Get local DOF indices for these joints
# dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

##########################
# オプション：制御ゲインの設定
##########################
# 位置ゲインの設定 (Kp)
# robot.set_dofs_kp(
#     kp             = np.array([20]*12),  # Example: 20 for each joint
#     dofs_idx_local = dofs_idx,
# )

# # 速度ゲインの設定 (Kd)
# robot.set_dofs_kv(
#     kv             = np.array([0.5]*12), # Example: 0.5 for each joint
#     dofs_idx_local = dofs_idx,
# )

# 安全のための力の範囲設定
# robot.set_dofs_force_range(
#     lower          = np.array([-20]*12), # Example: -20 for each joint
#     upper          = np.array([ 20]*12), # Example:  20 for each joint
#     dofs_idx_local = dofs_idx,
# )

##########################
# ハードリセット (例)
##########################
# for i in range(150):
#     if i < 50:
#         robot.set_dofs_position(
#             np.array([ 0.5,  0.5,  0.5,
#                        0.5,  0.5,  0.5,
#                        0.5,  0.5,  0.5,
#                        0.5,  0.5,  0.5]),
#             dofs_idx
#         )
#     elif i < 100:
#         robot.set_dofs_position(
#             np.array([-0.5, -0.5, -0.5,
#                       -0.5, -0.5, -0.5,
#                       -0.5, -0.5, -0.5,
#                       -0.5, -0.5, -0.5]),
#             dofs_idx
#         )
#     else:
#         robot.set_dofs_position(
#             np.zeros(12),  # all zeros
#             dofs_idx
#         )

#     scene.step()

##########################
# メイン制御ループ (PD制御など)
##########################
for i in range(125000):
    # if i == 0:
    #     # すべての関節にある程度の角度を与える
    #     robot.control_dofs_position(
    #         np.array([0.3, 0.3, 0.3,
    #                   0.3, 0.3, 0.3,
    #                   0.3, 0.3, 0.3,
    #                   0.3, 0.3, 0.3]),
    #         dofs_idx
    #     )
    # elif i == 250:
    #     # 全関節を別の位置へ
    #     robot.control_dofs_position(
    #         np.array([-0.3, -0.3, -0.3,
    #                   -0.3, -0.3, -0.3,
    #                   -0.3, -0.3, -0.3,
    #                   -0.3, -0.3, -0.3]),
    #         dofs_idx
    #     )
    # elif i == 500:
    #     # 全関節をゼロへ
    #     robot.control_dofs_position(
    #         np.zeros(12),
    #         dofs_idx
    #     )
    # elif i == 750:
    #     # 最初の関節を速度制御し、その他を位置制御 (例)
    #     robot.control_dofs_position(
    #         np.zeros(11),      # for joints [1:] (11 joints)
    #         dofs_idx[1:]
    #     )
    #     robot.control_dofs_velocity(
    #         np.array([1.0]),   # just the 1st joint
    #         dofs_idx[:1]
    #     )
    # elif i == 1000:
    #     # 全関節を力制御で0に
    #     robot.control_dofs_force(
    #         np.zeros(12),
    #         dofs_idx
    #     )

    # # これは与えられた制御コマンドに基づいて計算された制御力です
    # print("control force:", robot.get_dofs_control_force(dofs_idx))
    # # これは実際に関節が受けている内部力です
    # print("internal force:", robot.get_dofs_force(dofs_idx))
    time.sleep(0.3)
    scene.step()