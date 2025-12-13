import argparse
import genesis as gs
import numpy as np
import torch
import time

def get_height_at_xy(height_field, x, y, horizontal_scale, vertical_scale, center_x, center_y):
    # Convert world coordinates to heightfield indices
    # i = int(x / horizontal_scale)  # X to col
    # j = int(y  / horizontal_scale)  # Y to row 


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

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init()

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    ########################## parameters ##########################
    n = 2
    m = 2
    subterrain_size = 6.0
    horizontal_scale = 0.05
    vertical_scale = 0.005
    n_subterrains = (n, m)

    total_width = n_subterrains[0] * subterrain_size
    total_height = n_subterrains[1] * subterrain_size

    center_x = total_width / 2
    center_y = total_height / 2

    grid = [[None for _ in range(m)] for _ in range(n)]
    grid[0][0] = "pyramid_overhang_stairs_terrain"
    grid[0][1] = "pyramid_overhang_stairs_terrain"
    # grid[0][2] = "shallow_discrete_obstacles_terrain"
    grid[1][0] = "pyramid_sloped_terrain"
    grid[1][1] = "pyramid_down_sloped_terrain"
    # grid[1][2] = "pyramid_down_stairs_terrain"
    ########################## create terrain ##########################


    terrain =  gs.morphs.Terrain(
            pos=(-center_x, -center_y, 0),
            subterrain_size=(subterrain_size, subterrain_size),
            n_subterrains=n_subterrains,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            subterrain_types=grid,
        )

    global_terrain = scene.add_entity(terrain)
    # scene.add_entity(terrain)
    ball1 = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(1.0, 1.0, 1.0),
        ),
    )
    ball2 = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(1.0, 1.0, 1.0),
        ),
    )
    ball3 = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(1.0, 1.0, 1.0),
        ),
    )
    ball4 = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(1.0, 1.0, 1.0),
        ),
    )
    ball5 = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(1.0, 1.0, 1.0),
        ),
    )
    ball6 = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(1.0, 1.0, 1.0),
        ),
    )
    ########################## build ##########################
    scene.build()

    ########################## get heightfield ##########################
    height_field = global_terrain.geoms[0].metadata["height_field"]

    ########################## query height ##########################
    # x_query = 0.0
    # y_query = 0.0
    # z_query = get_height_at_xy(height_field, x_query, y_query, horizontal_scale, vertical_scale, center_x, center_y)
    # print(f"Height at ({x_query},{y_query}): {z_query}")

    # ########################## add a ball ##########################
    # ball.set_pos(torch.tensor((x_query, y_query, z_query + 0.4)))
    ########################## simulation loop ##########################
    balls = [ball1, ball2, ball3, ball4, ball5, ball6]
    k =0
    terrain_origin_x, terrain_origin_y, terrain_origin_z = terrain.pos
    print(f"terrain_origin_x & terrain_origin_y are ({terrain_origin_x},{terrain_origin_y})")
    print(height_field.shape)
    for i in range(n_subterrains[0]):   
        for j in range(n_subterrains[1]):       
            subterrain_center_x = terrain_origin_x + (i + 0.5) * subterrain_size
            subterrain_center_y = terrain_origin_y + (j + 0.5) * subterrain_size

            subterrain_center_z = get_height_at_xy(
                height_field,
                subterrain_center_x,
                subterrain_center_y,
                horizontal_scale,
                vertical_scale,
                center_x,
                center_y
            )
            balls[k].set_pos(torch.tensor((subterrain_center_x, subterrain_center_y, subterrain_center_z + 0.4)))
            print(f"Height at ({subterrain_center_x},{subterrain_center_y}): {subterrain_center_z}")
            k += 1

    # for i in range(height_field[0]):
    #     for j in range(height_field[1]):
    #         print(f"Height at ({i},{j}): {height_field[i, j] * vertical_scale}")


    for step in range(10000):
        scene.step()
        time.sleep(10)

if __name__ == "__main__":
    main()
