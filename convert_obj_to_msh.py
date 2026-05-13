import meshio
import sys
from pathlib import Path

obj_path = Path(sys.argv[1])
assert obj_path.suffix == ".obj"

msh_path = obj_path.with_suffix(".msh")

mesh = meshio.read(obj_path)
meshio.write(msh_path, mesh)