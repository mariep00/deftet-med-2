import argparse
import numpy as np
from simnibs import mesh_io
from pathlib import Path


def read_tet_npz(path):
    data = np.load(path)
    vertices = data["vertices"]
    tets = data["tets"]
    tet_occ = np.asarray(data["tet_occ"], dtype=np.float64).reshape(-1)

    print(data.files)
    print("vertices:", data["vertices"].shape, data["vertices"].dtype)
    print("tets:", data["tets"].shape, data["tets"].dtype)
    print("max tet index:", data["tets"].max())
    print("num vertices:", data["vertices"].shape[0])
    print("tet_occ:", tet_occ.shape, tet_occ.dtype)
    
    return vertices, tets, tet_occ


def vt_to_msh(vertices, tets, tet_occ, msh_out_path):
    vertices = np.asarray(vertices, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    tet_occ = np.asarray(tet_occ, dtype=np.float64).reshape(-1)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {vertices.shape}")

    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tets must be (F,4), got {tets.shape}")

    if tets.min() < 0 or tets.max() >= vertices.shape[0]:
        raise ValueError(
            f"Tet indices out of bounds: [{tets.min()}, {tets.max()}], "
            f"but N={vertices.shape[0]}"
        )
    
    if len(tet_occ) != len(tets):
        raise ValueError(
        f"tet_occ length ({len(tet_occ)}) must match number of tetrahedra ({len(tets)})"
    )

    # --- Write .msh ---
    m = mesh_io.Msh()
    #m.nodes = mesh_io.Nodes(vertices)
    m.elm = mesh_io.Elements(tetrahedra=tets + 1)  # convert to 1-based
    scale = 100.0  # or even 1000?
    m.nodes = mesh_io.Nodes(scale * vertices)
    m.add_element_field(tet_occ, "tet_occ")
    m.write(str(msh_out_path))

def convert_one(npz_path, out_path):
    vertices, tets, tet_occ = read_tet_npz(npz_path)
    vt_to_msh(vertices, tets, tet_occ, out_path)

    print("Conversion successful.")
    print("Vertices:", vertices.shape)
    print("Tets:", tets.shape)
    print("Tet occupancy:", tet_occ.shape)
    print("Output:", out_path)

'''def main():
    parser = argparse.ArgumentParser(description="Convert NPZ tetra mesh to .msh")
    parser.add_argument("--npz", required=True, help="Input .npz file")
    parser.add_argument("--out", required=True, help="Output .msh file")
    args = parser.parse_args()

    vertices, tets, tet_occ = read_tet_npz(args.npz)
    vt_to_msh(vertices, tets, tet_occ, args.out)

    print("Conversion successful.")
    print("Vertices:", vertices.shape)
    print("Tets:", tets.shape)
    print("Tet occupancy:", tet_occ.shape)
    print("Output:", args.out)'''

def main():
    parser = argparse.ArgumentParser(description="Convert NPZ tetra mesh to .msh")
    parser.add_argument("--npz", required=True, help="Input .npz file or folder")
    parser.add_argument("--out", required=True, help="Output .msh file or output folder")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --npz is a folder, search recursively"
    )

    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)

    if npz_path.is_file():
        if out_path.suffix.lower() != ".msh":
            raise ValueError("When --npz is a file, --out must be a .msh file")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        convert_one(npz_path, out_path)

    elif npz_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)

        pattern = "**/*.npz" if args.recursive else "*.npz"
        npz_files = sorted(npz_path.glob(pattern))

        if not npz_files:
            raise ValueError(f"No .npz files found in folder: {npz_path}")

        print(f"Found {len(npz_files)} .npz files.")

        for file_path in npz_files:
            if args.recursive:
                relative_path = file_path.relative_to(npz_path)
                msh_path = out_path / relative_path.with_suffix(".msh")
                msh_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                msh_path = out_path / f"{file_path.stem}.msh"

            convert_one(file_path, msh_path)

        print(f"Finished converting {len(npz_files)} files.")
        print("Output folder:", out_path)

    else:
        raise FileNotFoundError(f"Input path does not exist: {npz_path}")



if __name__ == "__main__":
    main()