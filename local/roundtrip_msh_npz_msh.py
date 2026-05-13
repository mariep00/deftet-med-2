import numpy as np
from simnibs import mesh_io


# ---------------------------------------------------
#  IO
# ---------------------------------------------------

def save_tet_(out_path, vertices, tets, vertices_dtype=np.float32):
    vertices = np.asarray(vertices, dtype=vertices_dtype)
    tets = np.asarray(tets, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {vertices.shape}")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tets must be (F,4), got {tets.shape}")

    np.savez(out_path, vertices=vertices, tets=tets)


def read_tet_(path):
    data = np.load(path)
    return data["vertices"], data["tets"]


# ---------------------------------------------------
# MSH <-> (V,T)
# ---------------------------------------------------

def msh_to_vt(msh_path):
    """
    Robustly read .msh and return:
      V: (N,3) float64
      T: (F,4) int64, 0-based
    """
    m = mesh_io.read_msh(msh_path)
    V = m.nodes.node_coord  # (N,3)

    # elm.node_number_list is typically (n_elements, max_nodes_per_element)
    # For tetrahedra, first 4 entries are node ids (1-based in .msh)
    nn = np.asarray(m.elm.node_number_list)

    # Identify tetra elements using elm.elm_type (Gmsh: 4-node tetra = type 4)
    # In SimNIBS, elm_type follows Gmsh element type codes.
    et = np.asarray(m.elm.elm_type).astype(int)

    T_1based = nn[et == 4, :4]  # keep only tetrahedra, first 4 nodes
    if T_1based.size == 0:
        raise ValueError(f"No tetrahedra (elm_type==4) found in {msh_path}")

    T = T_1based.astype(np.int64) - 1
    return V, T


def vt_to_msh(V, T, msh_out_path):
    """
    Write .msh from:
        V: (N,3)
        T: (F,4), 0-based
    """
    V = np.asarray(V, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)

    if T.min() < 0 or T.max() >= V.shape[0]:
        raise ValueError("Tet indices out of bounds.")

    m = mesh_io.Msh()
    m.nodes = mesh_io.Nodes(V)
    m.elm = mesh_io.Elements(tetrahedra=T + 1)  # back to 1-based
    m.write(msh_out_path)


# ---------------------------------------------------
# Comparison
# ---------------------------------------------------

def compare_meshes(V0, T0, V1, T1, atol=1e-6):
    if V0.shape != V1.shape:
        return False, "Vertex shape mismatch"

    if not np.allclose(V0, V1, atol=atol):
        return False, "Vertex coordinates differ"

    if T0.shape != T1.shape:
        return False, "Tet shape mismatch"

    # Compare as sets (ignore ordering)
    A = np.unique(np.sort(T0, axis=1), axis=0)
    B = np.unique(np.sort(T1, axis=1), axis=0)

    if A.shape != B.shape or not np.array_equal(A, B):
        return False, "Tet connectivity differs"

    return True, "Meshes match"


# ---------------------------------------------------
# Main Loop
# ---------------------------------------------------

def main():

    msh_input = "cube_test.msh"
    _file = "intermediate."
    msh_output = "reconstructed.msh"

    print("\n--- STEP 1: Read original .msh ---")
    V0, T0 = msh_to_vt(msh_input)
    print("Vertices:", V0.shape)
    print("Tets:", T0.shape)

    print("\n--- STEP 2: Save to . ---")
    save_tet_(_file, V0, T0)
    print("Saved:", _file)

    print("\n--- STEP 3: Load . ---")
    V_, T_ = read_tet_(_file)
    print("Loaded vertices:", V_.shape, V_.dtype)
    print("Loaded tets:", T_.shape, T_.dtype)

    print("\n--- STEP 4: Write reconstructed .msh ---")
    vt_to_msh(V_, T_, msh_output)
    print("Saved:", msh_output)

    print("\n--- STEP 5: Read reconstructed .msh ---")
    V1, T1 = msh_to_vt(msh_output)

    print("\n--- STEP 6: Compare ---")
    ok, msg = compare_meshes(V0, T0, V1, T1)
    print("Result:", ok, "-", msg)


if __name__ == "__main__":
    main()