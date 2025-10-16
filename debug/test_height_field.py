import argparse, genesis as gs, torch, time

# ---------- helper: world → (row, col) ----------
def height_index(x, y, terr, terrain_morph, horiz, center_x, center_y):
    local_x = x - terrain_morph.pos[0]          # world → local
    local_y = y - terrain_morph.pos[1]
    col = int(local_x / horiz)
    row = terr.geoms[0].metadata["height_field"].shape[0] - int(local_y / horiz) - 1
    return row, col

def get_height_at_xy(height_field, row, col, vert):
    return height_field[row, col] * vert
# ------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", default=False)
args = parser.parse_args()

# ------------- basic set‑up ---------------------
gs.init()
scene = gs.Scene(show_viewer=True)
n = 2
sub = 3.0
horiz = 0.05
vert  = 0.005
center_x = center_y = n*sub/2         # 3.0

grid = [
    ["flat_terrain",               "flat_terrain"              ],  # row‑0 (north)
    ["pyramid_steep_down_stairs_terrain", "pyramid_down_stairs_terrain"]  # row‑1 (south)
]

# terr = gs.morphs.Terrain(
#     pos=(-center_x, -center_y, 0),
#     subterrain_size=(sub, sub),
#     n_subterrains=(n, n),
#     horizontal_scale=horiz,
#     vertical_scale=vert,
#     subterrain_types=grid,
# )
# scene.add_entity(terr)
# scene.build()
# hf = terr.geoms[0].metadata["height_field"]

# 1.  create the morph (just geometry data, no physics yet)
terrain_morph = gs.morphs.Terrain(
    pos=(-center_x, -center_y, 0),
    subterrain_size=(sub, sub),
    n_subterrains=(n, n),
    horizontal_scale=horiz,
    vertical_scale=vert,
    subterrain_types=grid,
)

# 2.  add it to the scene → returns a RigidEntity
terrain_ent = scene.add_entity(terrain_morph)

# 3.  build the scene (creates geoms, bodies, etc.)
scene.build()

# 4.  now geoms is available
hf = terrain_ent.geoms[0].metadata["height_field"]


print("height_field shape", hf.shape)

# -------- test four sub‑terrain centres ----------
centres = [(-1.5,-1.5), ( 1.5,-1.5),
           (-1.5, 1.5), ( 1.5, 1.5)]

for x, y in centres:
    r, c = height_index(x, y, terrain_ent, terrain_morph, horiz, center_x, center_y)
    z = get_height_at_xy(hf, r, c, vert)
    # which (row,col) tile did we hit?
    tile_row = 1 if y < 0 else 0       # south == row‑1
    tile_col = 1 if x > 0 else 0
    print(f"({x:+4.1f},{y:+4.1f})  →  grid[{tile_row}][{tile_col}] "
          f"= {grid[tile_row][tile_col]:32s}  height {z:+.3f}")

# (viewer loop omitted for brevity)
