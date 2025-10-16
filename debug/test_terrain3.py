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

   
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/custom_plane_air.urdf", fixed=True),vis_mode="collision")
    scene.add_entity(gs.morphs.MJCF(file="xml/go1/go1.xml",pos=(0.0, 0.0, 0.55)))

    # for i in range(height_field[0]):
    #     for j in range(height_field[1]):
    #         print(f"Height at ({i},{j}): {height_field[i, j] * vertical_scale}")

    scene.build()
    for step in range(10000):
        scene.step()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
