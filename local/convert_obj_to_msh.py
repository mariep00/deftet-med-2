import sys
from pathlib import Path
import shutil
import subprocess
import meshio

SCALE = 100.0


def main():
    obj_path = Path(sys.argv[1])
    msh_path = obj_path.with_suffix(".msh")
    mesh = meshio.read(obj_path)
    # Match convert_tet_gmesh.py: scale coordinates up so Gmsh displays them clearly.
    mesh.points[:, :3] *= SCALE
    # Remove all extra data blocks that can create $NodeData/$ElementData
    mesh.point_data = {}
    mesh.cell_data = {}
    mesh.field_data = {}

    meshio.write(msh_path, mesh, file_format="gmsh22", binary=False)

    gmsh = shutil.which("gmsh")
    if gmsh is None:
        fallback = Path("/Users/mariepicquet/Applications/SimNIBS-4.5/bin/gmsh")
        if fallback.is_file():
            gmsh = str(fallback)
    if gmsh is not None:
        stl_path = obj_path.with_suffix(".stl")
        meshio.write(stl_path, mesh, binary=False)
        subprocess.run(
            [gmsh, str(stl_path), "-2", "-format", "msh2", "-o", str(msh_path), "-v", "2"],
            check=True)

    # Verification step
    mesh2 = meshio.read(msh_path)   # actually parse it
    print("Cell types:", mesh2.cells_dict.keys())
    print("Bounds:", mesh2.points.min(axis=0), mesh2.points.max(axis=0))



if __name__ == "__main__":
    main()
