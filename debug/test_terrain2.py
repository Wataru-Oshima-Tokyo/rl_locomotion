import argparse
import genesis as gs
import numpy as np
import torch
import time
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

   
    # scene.add_entity(gs.morphs.URDF(file="urdf/plane/custom_plane.urdf", fixed=True),vis_mode="collision")
    urdf_path = build_plane_urdf_with_step(0.3)  # 絶対パスを返す

    scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True))
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
