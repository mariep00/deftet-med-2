import sys
import torch
from pathlib import Path

repo_dir = Path(__file__).resolve().parent
workspace_root = repo_dir.parent
sys.path.append(str(repo_dir))

from dataloader import create_dataloader

mesh_surface_dir = workspace_root / "mesh_surfaces"
cache_root = repo_dir / "dataset_cache" / "debug_run"

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "No GPU detected. Run this on a GPU node."

dl = create_dataloader(
    msh_source=str(mesh_surface_dir),
    save_cache_root=str(cache_root),
    train=False,
    batch_size=1,
    only_chairs=False,
)

print("Dataloader created")

batch = next(iter(dl))

print("Loaded one batch successfully")
print("keys:", batch.keys())
print("num meshes:", len(batch["verts"]))
print("sample_points shape:", batch["sample_points"].shape)
print("sdf_point shape:", batch["sdf_point"].shape)
print("sdf_value shape:", batch["sdf_value"].shape)
print("name:", batch["name"][0])
print("synset:", batch["synset"][0])
