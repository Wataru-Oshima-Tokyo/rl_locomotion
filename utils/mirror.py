def mirror_not_changed_values(values):
    return values

def mirror_changed_minus_values(values):
    values_mirrored = -values
    return values_mirrored

def mirror_linear(linear):
    linear_mirrored = linear.clone()
    linear_mirrored[..., 1] *= -1 # y
    return linear_mirrored

def mirror_angle(angle):
    angle_mirrored = angle.clone()
    angle_mirrored[..., 0] *= -1 # roll
    angle_mirrored[..., 2] *= -1 # yaw
    return angle_mirrored

def mirror_commands(commands):
    commands_mirrored = commands.clone()
    commands_mirrored[..., 1] *= -1 # y
    commands_mirrored[..., 2] *= -1 # yaw
    return commands_mirrored

def mirror_go_leg_joint(leg):
    leg_mirrored = leg.clone()
    leg_mirrored[..., [0, 1, 2]] = leg[..., [3, 4, 5]]
    leg_mirrored[..., [3, 4, 5]] = leg[..., [0, 1, 2]]
    leg_mirrored[..., [6, 7, 8]] = leg[..., [9, 10, 11]]
    leg_mirrored[..., [9, 10, 11]] = leg[..., [6, 7, 8]]
    leg_mirrored[..., 0::3] = -leg_mirrored[..., 0::3] # hip joint
    return leg_mirrored

def mirror_go_foot(foot):
    foot_mirrored = foot.clone()
    foot_mirrored[..., 0] = foot[..., 1]
    foot_mirrored[..., 1] = foot[..., 0]
    foot_mirrored[..., 2] = foot[..., 3]
    foot_mirrored[..., 3] = foot[..., 2]
    return foot_mirrored

def mirror_hg_leg_joint(leg):
    leg_mirrored = leg.clone()
    leg_mirrored[..., [0, 1, 2, 3, 4, 5]] = leg[..., [6, 7, 8, 9, 10, 11]]
    leg_mirrored[..., [6, 7, 8, 9, 10, 11]] = leg[..., [0, 1, 2, 3, 4, 5]]
    leg_mirrored[..., [1, 2, 5]] = -leg_mirrored[..., [1, 2, 5]]   # leg roll, yaw joint
    leg_mirrored[..., [7, 8, 11]] = -leg_mirrored[..., [7, 8, 11]] # leg roll, yaw joint
    return leg_mirrored

def mirror_hg_waist_1_joint(waist):
    waist_mirrored = -waist # waist yaw
    return waist_mirrored

def mirror_hg_waist_3_joint(waist):
    waist_mirrored = waist.clone()
    waist_mirrored[..., 0] *= -1 # waist yaw
    waist_mirrored[..., 1] *= -1 # waist roll
    return waist_mirrored

def mirror_hg_arm_5_joint(arm):
    arm_mirrored = arm.clone()
    arm_mirrored[..., [0, 1, 2, 3, 4]] = arm[..., [5, 6, 7, 8, 9]]
    arm_mirrored[..., [5, 6, 7, 8, 9]] = arm[..., [0, 1, 2, 3, 4]]
    arm_mirrored[..., [1, 2, 4]] = -arm_mirrored[..., [1, 2, 4]] # arm roll, yaw joint
    arm_mirrored[..., [6, 7, 9]] = -arm_mirrored[..., [6, 7, 9]] # arm roll, yaw joint
    return arm_mirrored

def mirror_hg_arm_7_joint(arm):
    arm_mirrored = arm.clone()
    arm_mirrored[..., [0, 1, 2, 3, 4, 5, 6]] = arm[..., [7, 8, 9, 10, 11, 12, 13]]
    arm_mirrored[..., [7, 8, 9, 10, 11, 12, 13]] = arm[..., [0, 1, 2, 3, 4, 5, 6]]
    arm_mirrored[..., [1, 2, 4, 6]] = -arm_mirrored[..., [1, 2, 4, 6]]     # arm roll, yaw joint
    arm_mirrored[..., [8, 9, 11, 13]] = -arm_mirrored[..., [8, 9, 11, 13]] # arm roll, yaw joint
    return arm_mirrored

def mirror_hg_23_joint(joint):
    joint_mirrored = joint.clone()
    joint_mirrored[..., [0, 1, 2, 3, 4, 5]] = joint[..., [6, 7, 8, 9, 10, 11]]   # leg
    joint_mirrored[..., [6, 7, 8, 9, 10, 11]] = joint[..., [0, 1, 2, 3, 4, 5]]   # leg
    joint_mirrored[..., [13, 14, 15, 16, 17]] = joint[..., [18, 19, 20, 21, 22]] # arm
    joint_mirrored[..., [18, 19, 20, 21, 22]] = joint[..., [13, 14, 15, 16, 17]] # arm
    joint_mirrored[..., [1, 2, 5]] = -joint_mirrored[..., [1, 2, 5]]             # leg roll, yaw joint
    joint_mirrored[..., [7, 8, 11]] = -joint_mirrored[..., [7, 8, 11]]           # leg roll, yaw joint
    joint_mirrored[..., 12] = -joint_mirrored[..., 12]                           # waist yaw joint
    joint_mirrored[...,[14, 15, 17]] = -joint_mirrored[...,[14, 15, 17]]         # arm roll, yaw joint 
    joint_mirrored[...,[19, 20, 22]] = -joint_mirrored[...,[19, 20, 22]]         # arm roll, yaw joint
    return joint_mirrored

def mirror_hg_27_joint(joint):
    joint_mirrored = joint.clone()
    joint_mirrored[..., [0, 1, 2, 3, 4, 5]] = joint[..., [6, 7, 8, 9, 10, 11]]                   # leg
    joint_mirrored[..., [6, 7, 8, 9, 10, 11]] = joint[..., [0, 1, 2, 3, 4, 5]]                   # leg
    joint_mirrored[..., [13, 14, 15, 16, 17, 18, 19]] = joint[..., [20, 21, 22, 23, 24, 25, 26]] # arm
    joint_mirrored[..., [20, 21, 22, 23, 24, 25, 26]] = joint[..., [13, 14, 15, 16, 17, 18, 19]] # arm
    joint_mirrored[..., [1, 2, 5]] = -joint_mirrored[..., [1, 2, 5]]                             # leg roll, yaw joint
    joint_mirrored[..., [7, 8, 11]] = -joint_mirrored[..., [7, 8, 11]]                           # leg roll, yaw joint
    joint_mirrored[..., 12] = -joint_mirrored[..., 12]                                           # waist yaw joint
    joint_mirrored[...,[14, 15, 17, 19]] = -joint_mirrored[...,[14, 15, 17, 19]]                 # arm roll, yaw joint 
    joint_mirrored[...,[21, 22, 24, 26]] = -joint_mirrored[...,[21, 22, 24, 26]]                 # arm roll, yaw joint
    return joint_mirrored

def mirror_hg_29_joint(joint):
    joint_mirrored = joint.clone()
    joint_mirrored[..., [0, 1, 2, 3, 4, 5]] = joint[..., [6, 7, 8, 9, 10, 11]]                   # leg
    joint_mirrored[..., [6, 7, 8, 9, 10, 11]] = joint[..., [0, 1, 2, 3, 4, 5]]                   # leg
    joint_mirrored[..., [15, 16, 17, 18, 19, 20, 21]] = joint[..., [22, 23, 24, 25, 26, 27, 28]] # arm
    joint_mirrored[..., [22, 23, 24, 25, 26, 27, 28]] = joint[..., [15, 16, 17, 18, 19, 20, 21]] # arm
    joint_mirrored[..., [1, 2, 5]] = -joint_mirrored[..., [1, 2, 5]]                             # leg roll, yaw joint
    joint_mirrored[..., [7, 8, 11]] = -joint_mirrored[..., [7, 8, 11]]                           # leg roll, yaw joint
    joint_mirrored[..., [12, 13]] = -joint_mirrored[..., [12, 13]]                               # waist yaw, roll joint
    joint_mirrored[...,[16, 17, 19, 21]] = -joint_mirrored[...,[16, 17, 19, 21]]                 # arm roll, yaw joint 
    joint_mirrored[...,[23, 24, 26, 28]] = -joint_mirrored[...,[23, 24, 26, 28]]                 # arm roll, yaw joint
    return joint_mirrored

def mirror_hg_foot(foot):
    foot_mirrored = foot.clone()
    foot_mirrored[..., 0] = foot[..., 1]
    foot_mirrored[..., 1] = foot[..., 0]
    return foot_mirrored