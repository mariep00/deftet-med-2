'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import argparse
import csv
import json
import math
import os
import warnings
from collections import defaultdict

import kaolin as kal
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import OPTIONS
from dataloader import create_dataloader
from layers.DefTet.deftet import DefTet
from layers.pc_model import DeformableTetNetwork
from parallel import ParallelWrapper
from utils import dataloder_helper as helpers
from utils import tet_utils
from utils.experiment import Experiment
from utils.matrix_utils import MySparse
from utils.mesh_utils import save_mesh
from utils.point_cloud_utils import (
    chamfer_distance,
    chamfer_distance_l1,
    f_score,
    hausdorff_distance,
    iou as point_cloud_iou,
)

warnings.simplefilter("ignore", UserWarning)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = "posthoc_eval"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation for medical surface-mesh DefTet runs."
    )
    parser.add_argument("--experiment_path", required=True,
                        help="Path to a saved training run folder.")
    parser.add_argument("--dataset_dir", default=None,
                        help="Medical .msh dataset folder. Defaults to config.dataset_dir.")
    parser.add_argument("--split_file", default=None,
                        help="Split file to evaluate. Defaults to config.test_split_file, then val_split_file.")
    parser.add_argument("--checkpoint", choices=["best", "recent"], default="best",
                        help="Which checkpoint family to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Tet occupancy threshold used for predicted surface extraction.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size. Use 1 for safest per-mesh metrics.")
    parser.add_argument("--num_surface_samples", type=int, default=100000,
                        help="Number of points sampled from each predicted surface for metrics.")
    parser.add_argument("--f_score_radius", type=float, default=0.01,
                        help="Radius used by the F-score metric.")
    parser.add_argument("--hash_resolution", type=int, default=512,
                        help="Hash resolution for kaolin mesh occupancy checks.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Optional cap for smoke tests. 0 evaluates the full split.")
    parser.add_argument("--loader_workers", type=int, default=None,
                        help="Override dataloader workers. Defaults to config.loader_workers.")
    parser.add_argument("--cache_root", default=None,
                        help="Dataset preprocessing cache root. Defaults to a shared eval cache beside runs.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory. Relative paths are created inside the experiment folder.")
    parser.add_argument("--save_surfaces", action="store_true",
                        help="Save predicted and GT OBJ surfaces for inspected samples.")
    return parser.parse_args()


def expand_path(path):
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


def get_eval_output_dir(experiment, output_dir):
    output_dir = os.path.expanduser(output_dir)
    if os.path.isabs(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return experiment.dir_path(output_dir)


def get_n_point(config):
    if getattr(config, "input_points", 0) > 0:
        return config.input_points
    return 10000 if config.res >= 100 else 5000


def tensor_item(value, sample_idx=0, batch_size=1):
    if value is None:
        return math.nan
    if not torch.is_tensor(value):
        return float(value)

    value = value.detach()
    if value.numel() == 0:
        return math.nan
    flat = value.reshape(-1)
    if flat.numel() == batch_size:
        return float(flat[sample_idx].item())
    return float(value.float().mean().item())


def safe_float(value):
    if torch.is_tensor(value):
        value = value.detach().item()
    value = float(value)
    if math.isfinite(value):
        return value
    return math.nan


def summarize_rows(rows, metric_names):
    summary = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in rows
            if metric_name in row and row[metric_name] != "" and math.isfinite(float(row[metric_name]))
        ]
        if not values:
            summary[metric_name] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue

        arr = np.asarray(values, dtype=np.float64)
        summary[metric_name] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


class MedicalEvalEngine(object):
    def __init__(self, config, experiment, output_dir):
        self.config = config
        self.experiment = experiment
        self.output_dir = output_dir
        self.deftet = DefTet()

        if str(self.config.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this run config, but torch.cuda.is_available() is False.")

        vertices_nx3, tetrahedron_fx4, mask = helpers.read_tetrahedron(
            res=self.config.res,
            root=ROOT_DIR,
        )
        self.init_tet_pos = torch.from_numpy(vertices_nx3).to(
            self.config.device
        ) - 0.5
        self.init_pos_mask = torch.from_numpy(mask).float().to(self.config.device)
        self.init_tet_fx4 = torch.from_numpy(tetrahedron_fx4).long().to(self.config.device)

        point_adj_sparse = tet_utils.c_tet_to_adj_sparse(
            vertices_nx3,
            tetrahedron_fx4,
            normalize=True,
        ).to(self.config.device)
        self.point_adj_sparse = MySparse(point_adj_sparse)

        tet_face_fx3, tet_facetet_idx_fx2, _, _ = tet_utils.tet_to_face(
            vertices_nx3.shape[0],
            tetrahedron_fx4,
        )
        self.tet_face_fx3 = torch.from_numpy(tet_face_fx3).long().to(self.config.device)
        self.tet_face_tetidx_fx2 = torch.from_numpy(tet_facetet_idx_fx2).long().to(self.config.device)

        self.model = DeformableTetNetwork(
            self.config.device,
            scale_pos=self.config.scale_pos,
            train_def=not (self.config.lambda_def == 0.),
            point_cloud=self.config.point_cloud,
            point_adj_sparse=self.point_adj_sparse,
            use_graph_attention=self.config.use_graph_attention,
            upscale=self.config.upscale,
            use_two_encoder=self.config.use_two_encoder,
            timing=self.config.timing,
            use_lap_layer=self.config.use_lap_layer,
            use_disn=self.config.use_disn,
            scale_pvcnn=self.config.scale_pvcnn,
        )

        inverse_v = nn.Parameter(self.deftet.tet_inverse_v(
            self.init_tet_pos,
            self.init_tet_fx4,
        ))
        inverse_v.requires_grad = False
        self.deftet.inverse_v = inverse_v.to(self.config.device)

        self.parallel = ParallelWrapper(
            self.model,
            self.deftet,
            os.path.join(self.output_dir, "visualization"),
            self.point_adj_sparse,
            n_all_device=1,
            timing=None,
            use_two_encoder=self.config.use_two_encoder,
            add_input_noise=self.config.add_input_noise,
            n_point=get_n_point(self.config),
            use_lap_layer=self.config.use_lap_layer,
            use_point=self.config.point_cloud,
        )

    def load_checkpoint(self, checkpoint):
        prefix = "best_" if checkpoint == "best" else "recent_"
        checkpoint_specs = [
            ("encoder", self.model.encoder),
            ("decoder_occ", self.model.decoder_occ),
            ("decoder_pos", self.model.decoder_pos),
        ]
        if self.config.use_lap_layer:
            checkpoint_specs.append(("lap_decoder_pos", self.model.lap_decoder_pos))

        for name, module in checkpoint_specs:
            path = os.path.join(self.experiment.root_path, prefix + name + ".pth")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing checkpoint file: {path}")
            state_dict = torch.load(path, map_location=self.config.device)
            module.load_state_dict(state_dict)

    def evaluate(self, dataloader, args):
        self.model.eval()
        self.parallel.training = False
        rows = []
        metric_names = [
            "iou",
            "f_score",
            "chamfer",
            "chamfer_l1",
            "mean_hausdorff",
            "max_hausdorff",
            "surf",
            "occ_iou",
            "occ_loss",
            "lap",
            "edge",
            "area",
            "delta",
            "amips",
            "surf_chamfer",
        ]

        surfaces_dir = os.path.join(self.output_dir, "surfaces")
        if args.save_surfaces:
            os.makedirs(surfaces_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(dataloader, 0), desc="eval"):
                if args.max_samples > 0 and len(rows) >= args.max_samples:
                    break

                imgs = data["imgs"][:, :3].float().to(self.config.device) if "imgs" in data else None
                points = data["sdf_point"].float().to(self.config.device)
                gt_occ = data["sdf_value"].float().to(self.config.device)
                surface_point = data["sample_points"].float().to(self.config.device)
                cam_rot = data["cam_rot"].float().to(self.config.device) if "cam_rot" in data else None
                cam_pos = data["cam_pos"].float().to(self.config.device) if "cam_pos" in data else None
                cam_proj = data["cam_proj"].float().to(self.config.device) if "cam_proj" in data else None

                batch_size = surface_point.shape[0]
                all_verts = [
                    verts.to(self.config.device).unsqueeze(0).expand(1, -1, -1)
                    for verts in data["verts"]
                ]
                all_faces = [
                    faces.to(self.config.device).unsqueeze(0).expand(1, -1, -1)
                    for faces in data["faces"]
                ]

                init_tet_pos_bxnx3 = self.init_tet_pos.float().unsqueeze(
                    0
                ).expand(batch_size, -1, -1)
                init_tet_pos_mask = self.init_pos_mask.float().unsqueeze(
                    0
                ).expand(batch_size, -1, -1)
                if not self.config.use_init_pos_mask:
                    init_tet_pos_mask = None
                init_tet_bxfx4 = self.init_tet_fx4.unsqueeze(
                    0
                ).expand(batch_size, -1, -1)
                tet_face_tetidx_bxfx2 = self.tet_face_tetidx_fx2.unsqueeze(
                    0
                ).expand(batch_size, -1, -1)
                init_tet_face_bxfx3 = self.tet_face_fx3.unsqueeze(
                    0
                ).expand(batch_size, -1, -1)

                amips_energy, edge, area_variance, surface_align, normal_loss, \
                    occ_loss, occ_iou, lap, delta_loss, tet_pos, pred_occ_prob, condition, \
                    surface, pred_surface, other_chamfer_distance, _ = self.parallel(
                        imgs=imgs,
                        init_tet_pos_bxnx3=init_tet_pos_bxnx3,
                        init_tet_bxfx4=init_tet_bxfx4,
                        points=points,
                        surface_point=surface_point,
                        save=False,
                        global_step=batch_idx,
                        tet_face_tetidx_bxfx2=tet_face_tetidx_bxfx2,
                        all_verts=all_verts,
                        all_faces=all_faces,
                        return_all=True,
                        return_surf=True,
                        inference=True,
                        tet_face_bxfx3=init_tet_face_bxfx3,
                        init_pos_mask=init_tet_pos_mask,
                        cam_pos=cam_pos,
                        cam_rot=cam_rot,
                        cam_proj=cam_proj,
                        pred_threshold=args.threshold,
                        random_seed=batch_idx,
                    )

                for sample_idx, name in enumerate(data["name"]):
                    if args.max_samples > 0 and len(rows) >= args.max_samples:
                        break

                    row = {
                        "name": str(name),
                        "synset": str(data["synset"][sample_idx]),
                        "checkpoint": args.checkpoint,
                        "threshold": args.threshold,
                        "status": "ok",
                        "surf": tensor_item(surface_align, sample_idx, batch_size),
                        "occ_iou": tensor_item(occ_iou, sample_idx, batch_size),
                        "occ_loss": tensor_item(occ_loss, sample_idx, batch_size),
                        "lap": tensor_item(lap, sample_idx, batch_size),
                        "edge": tensor_item(edge, sample_idx, batch_size),
                        "area": tensor_item(area_variance, sample_idx, batch_size),
                        "delta": tensor_item(delta_loss, sample_idx, batch_size),
                        "amips": tensor_item(amips_energy, sample_idx, batch_size),
                        "surf_chamfer": tensor_item(other_chamfer_distance, sample_idx, batch_size),
                    }

                    pred_faces = pred_surface[sample_idx]
                    if pred_faces.shape[0] == 0:
                        row.update({
                            "status": "empty_surface",
                            "iou": math.nan,
                            "f_score": math.nan,
                            "chamfer": math.nan,
                            "chamfer_l1": math.nan,
                            "mean_hausdorff": math.nan,
                            "max_hausdorff": math.nan,
                        })
                        rows.append(row)
                        continue

                    mesh_v = tet_pos[sample_idx, pred_faces.reshape(-1)]
                    mesh_f = torch.arange(
                        0,
                        mesh_v.shape[0],
                        device=mesh_v.device,
                        dtype=torch.long,
                    ).reshape(-1, 3)

                    query_points = points[sample_idx:sample_idx + 1]
                    gt_occ_binary = (gt_occ[sample_idx:sample_idx + 1] > 0.0).float()
                    pred_occ_binary = kal.ops.mesh.check_sign(
                        mesh_v.unsqueeze(0),
                        mesh_f,
                        query_points,
                        hash_resolution=args.hash_resolution,
                    ).float()
                    row["iou"] = safe_float(point_cloud_iou(
                        pred_occ_binary,
                        gt_occ_binary,
                        thresh=0.5,
                    ))

                    pred_points, _ = kal.ops.mesh.sample_points(
                        mesh_v.unsqueeze(dim=0),
                        mesh_f,
                        args.num_surface_samples,
                    )
                    gt_surface_points = surface_point[sample_idx:sample_idx + 1]
                    row["f_score"] = safe_float(f_score(
                        gt_surface_points,
                        pred_points,
                        radius=args.f_score_radius,
                        extend=True,
                    ))
                    row["chamfer"] = safe_float(chamfer_distance(
                        gt_surface_points,
                        pred_points,
                    ))
                    row["chamfer_l1"] = safe_float(chamfer_distance_l1(
                        gt_surface_points,
                        pred_points,
                    ))
                    mean_hausdorff, max_hausdorff = hausdorff_distance(
                        mesh_v,
                        mesh_f,
                        data["verts"][sample_idx].to(self.config.device),
                        data["faces"][sample_idx].to(self.config.device),
                        pred_points[0],
                        surface_point[sample_idx],
                    )
                    row["mean_hausdorff"] = safe_float(mean_hausdorff)
                    row["max_hausdorff"] = safe_float(max_hausdorff)

                    if args.save_surfaces:
                        base_name = os.path.basename(str(name))
                        save_mesh(
                            mesh_v.detach().cpu().numpy(),
                            mesh_f.detach().cpu().numpy(),
                            os.path.join(surfaces_dir, f"{base_name}_pred.obj"),
                        )
                        save_mesh(
                            data["verts"][sample_idx].detach().cpu().numpy(),
                            data["faces"][sample_idx].detach().cpu().numpy(),
                            os.path.join(surfaces_dir, f"{base_name}_gt.obj"),
                        )

                    rows.append(row)

        csv_path = os.path.join(self.output_dir, "per_sample_metrics.csv")
        fieldnames = [
            "name",
            "synset",
            "checkpoint",
            "threshold",
            "status",
        ] + metric_names
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        aggregate = {
            "experiment_path": self.experiment.root_path,
            "checkpoint": args.checkpoint,
            "threshold": args.threshold,
            "num_rows": len(rows),
            "num_ok": sum(1 for row in rows if row["status"] == "ok"),
            "metrics": summarize_rows(rows, metric_names),
        }
        json_path = os.path.join(self.output_dir, "aggregate_metrics.json")
        with open(json_path, "w") as f:
            json.dump(aggregate, f, indent=2, sort_keys=True)

        print(f"Wrote per-sample metrics: {csv_path}")
        print(f"Wrote aggregate metrics: {json_path}")


def main():
    args = get_parser()
    args.experiment_path = expand_path(args.experiment_path)
    args.dataset_dir = expand_path(args.dataset_dir)
    args.split_file = expand_path(args.split_file)
    args.cache_root = expand_path(args.cache_root)

    experiment = Experiment.load(args.experiment_path, options=OPTIONS)
    config = experiment.config

    dataset_dir = args.dataset_dir or config.dataset_dir
    split_file = args.split_file or config.test_split_file or config.val_split_file or None
    if split_file is None:
        raise ValueError("No split file provided and config has no test_split_file or val_split_file.")

    cache_root = args.cache_root
    if cache_root is None:
        cache_root = os.path.join(os.path.dirname(experiment.root_path), "dataset_cache_eval")

    loader_workers = args.loader_workers
    if loader_workers is None:
        loader_workers = getattr(config, "loader_workers", 4)

    output_dir = get_eval_output_dir(experiment, args.output_dir)
    dataloader = create_dataloader(
        msh_source=dataset_dir,
        save_cache_root=cache_root,
        batch_size=args.batch_size,
        train=False,
        only_chairs=False,
        split_file=split_file,
        split_name="eval",
        num_workers=loader_workers,
    )

    engine = MedicalEvalEngine(config, experiment, output_dir)
    engine.load_checkpoint(args.checkpoint)
    engine.evaluate(dataloader, args)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
