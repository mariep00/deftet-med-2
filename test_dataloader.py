import sys
import torch
sys.path.append("/Users/mariepicquet/thesis/deftet-med")

from dataloader import create_dataloader

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "No GPU detected. Run this on a GPU node."

dl = create_dataloader(
    shapenet_source="/work3/s233736/datasets/shapenetcore",
    save_cache_root="/work3/s233736/deftet_runs/debug_run",
    train=False,
    batch_size=1,
    only_chairs=True,
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