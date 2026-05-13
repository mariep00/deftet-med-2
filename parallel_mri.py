'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

from torch import nn
import torch.nn.functional as F
import kaolin as kal
import torch
import os
from utils.point_cloud_utils import iou
EPS = 1e-10

def get_occupancy_function(verts, faces, device='cuda'):
    """
    NOTE: Only support batch size 1.
    """
    if len(faces.shape) >= 3 and faces.shape[0] > 1:
        raise ValueError('Batch size bigger than 1 is not supported')

    p1 = torch.index_select(verts, 0, faces[:, 0]).view(
        1, -1, 3).float().contiguous().to(device)
    p2 = torch.index_select(verts, 0, faces[:, 1]).view(
        1, -1, 3).float().contiguous().to(device)
    p3 = torch.index_select(verts, 0, faces[:, 2]).view(
        1, -1, 3).float().contiguous().to(device)

    def eval_query(query, intersection_function):
        return intersection_function(
            query.view(1, -1, 3).float(),
            p1.to(query.device),
            p2.to(query.device),
            p3.to(query.device)
        )[0]

    return eval_query

class TensorList(nn.Module):
    def __init__(self, t_list):
        super(TensorList, self).__init__()
        for i in range(len(t_list)):
            self.register_buffer('item_%d' % (i), t_list[i].cuda())

    def __getitem__(self, item):
        attr_name = 'item_%d' % (item)
        return getattr(self, attr_name)


class SimpleParallelWrapper(nn.Module):
    """
    Point-cloud-only version of ParallelWrapper.

    Removed:
    - image input support
    - use_point flag and image/point branching
    - camera-related arguments (cam_pos, cam_rot, cam_proj)

    Kept:
    - point-cloud encoding
    - optional input noise
    - optional dual encoder support
    - training/inference branching
    - occupancy thresholding
    - forward_surface_align
    - occupancy loss
    - IoU computation in inference
    - laplacian regularization
    - delta loss
    - optional return modes
    - save/global_step controls
    """

    def __init__(
        self,
        model,
        deftet,
        visualization_path,
        point_adj_sparse,
        n_all_device=1,
        n_point=5000,
        timing=None,
        use_two_encoder=False,
        add_input_noise=False,
        use_lap_layer=False,
    ):
        super().__init__()
        self.model = model
        self.deftet = deftet
        self.visualization_path = visualization_path
        self.point_adj_sparse = point_adj_sparse
        self.point_adj_sparse_list = {}
        self.n_all_device = n_all_device
        self.n_point = n_point
        self.timing = timing
        self.use_two_encoder = use_two_encoder
        self.add_input_noise = add_input_noise
        self.use_lap_layer = use_lap_layer

    def get_point_adj_sparse(self, device):
        if device not in self.point_adj_sparse_list:
            self.point_adj_sparse_list[device] = self.point_adj_sparse.construct().to(device)
        return self.point_adj_sparse_list[device]

    def forward(
        self,
        init_tet_pos_bxnx3,
        init_tet_bxfx4,
        points,
        surface_point,
        save=False,
        global_step=0,
        tet_face_tetidx_bxfx2=None,
        all_verts=None,
        all_faces=None,
        return_all=False,
        inference=False,
        return_surf=False,
        tet_face_bxfx3=None,
        init_pos_mask=None,
        pred_threshold=0.4,
        return_offset=False,
        random_seed=1,
    ):
        sum_time = 0
        has_gt_mesh = all_verts is not None and all_faces is not None


        # --------------------------------------------------
        # Encode point-cloud input
        # --------------------------------------------------
        if self.add_input_noise:
            if not self.training:
                torch.cuda.manual_seed(random_seed + 1)
                torch.random.manual_seed(random_seed)
            noise = torch.zeros_like(surface_point[:, :self.n_point]).normal_()
            noise = noise * 0.005
            input_points = surface_point[:, :self.n_point] + noise
        else:
            input_points = surface_point[:, :self.n_point]

        encoding = self.model.encode_inputs(input_points)
        z = None

        if self.use_two_encoder:
            encoding_pos = encoding[0]
            encoding_occ = encoding[1]
        else:
            encoding_pos = encoding
            encoding_occ = encoding

        # --------------------------------------------------
        # Decode tetrahedral positions
        # --------------------------------------------------
        pred_pos_delta, tet_pos, ori_pos_delta = self.model.decode_pos(
            init_tet_pos_bxnx3,
            z,
            encoding_pos,
            init_pos_mask,
        )

        tet_pos_for_occ = tet_pos

        # --------------------------------------------------
        # Predict occupancy for surface extraction
        # --------------------------------------------------
        if inference or self.use_lap_layer:
            with torch.no_grad():
                all_pred_occ_prob = self.model.split_decode_occ(
                    tet_pos,
                    z,
                    encoding_occ,
                    init_tet_bxfx4,
                )
                pred_occ_in_out = all_pred_occ_prob > pred_threshold
        else:
            pred_occ_in_out = None
            all_pred_occ_prob = None

        # --------------------------------------------------
        # Select mesh chunk for current GPU, if GT mesh exists
        # --------------------------------------------------
        if has_gt_mesh:
            curr_device_id = torch.cuda.current_device()
            n_all = len(all_verts)
            n_each = int(n_all / self.n_all_device)

            curr_verts = all_verts[n_each * curr_device_id : n_each * (curr_device_id + 1)]
            curr_faces = all_faces[n_each * curr_device_id : n_each * (curr_device_id + 1)]
            mesh_list = [curr_verts, curr_faces]
        else:
            mesh_list = None

        if save:
            print('Pos Delta min:', pred_pos_delta[-1].min(dim=0)[0])
            print('Pos Delta max:', pred_pos_delta[-1].max(dim=0)[0])
            print('Tet Pos min:', tet_pos[-1].min(dim=0)[0])
            print('Tet Pos max:', tet_pos[-1].max(dim=0)[0])

        # --------------------------------------------------
        # Surface alignment / geometric losses
        # --------------------------------------------------
        if inference:
            (
                amips_energy,
                edge,
                area_variance,
                surface_align,
                normal_loss,
                center_occ,
                condition,
                surface,
                pred_surface,
                other_chamfer_distance,
            ) = self.deftet.forward_surface_align(
                tet_pos,
                points,
                init_tet_bxfx4,
                mesh_list,
                gt_surface_points=surface_point,
                save=save,
                save_name=os.path.join(self.visualization_path, f'vis_{global_step}'),
                tet_face_tet_bx4fx2=tet_face_tetidx_bxfx2,
                inference=True,
                tet_face_bxfx3=tet_face_bxfx3,
                pred_occ=pred_occ_in_out,
            )
            lap_v_loss = torch.zeros_like(amips_energy)
        else:
            (
                amips_energy,
                edge,
                area_variance,
                surface_align,
                normal_loss,
                center_occ,
                surface,
                other_chamfer_distance,
                lap_v_loss,
            ) = self.deftet.forward_surface_align(
                tet_pos,
                points,
                init_tet_bxfx4,
                mesh_list,
                gt_surface_points=surface_point,
                save=save,
                save_name=os.path.join(self.visualization_path, f'vis_{global_step}'),
                tet_face_tet_bx4fx2=tet_face_tetidx_bxfx2,
                inference=False,
                tet_face_bxfx3=tet_face_bxfx3,
                pred_occ=pred_occ_in_out,
            )

        # --------------------------------------------------
        # Occupancy prediction at tet centers for BCE loss
        # --------------------------------------------------
        pred_occ, center_idx = self.model.decode_occ(
            tet_pos_for_occ,
            z,
            encoding_occ,
            init_tet_bxfx4,
        )

        pred_occ_logit = pred_occ.logits
        pred_occ_prob = pred_occ.probs

        if center_occ is not None:
            gt_occ = center_occ[:, center_idx]
            occ_loss = F.binary_cross_entropy_with_logits(
                pred_occ_logit,
                gt_occ
            ).mean()
        else:
            gt_occ = None
            occ_loss = torch.zeros((), device=tet_pos.device, dtype=tet_pos.dtype)

        # --------------------------------------------------
        # Regularization losses
        # --------------------------------------------------
        delta_loss = torch.mean(torch.abs(pred_pos_delta), dim=-1).mean(dim=-1)

        point_adj = self.get_point_adj_sparse(pred_pos_delta.device)
        lap = self.deftet.laplacian_sparse(pred_pos_delta, point_adj)

        # --------------------------------------------------
        # IoU during inference
        # --------------------------------------------------
        if inference:
            if gt_occ is not None:
                occ_iou = torch.tensor([
                    iou(pred_occ_logit[i], gt_occ[i], thresh=0.1)
                    for i in range(gt_occ.shape[0])
                ], device=gt_occ.device)
            else:
                occ_iou = torch.zeros(
                    tet_pos.shape[0],
                    device=tet_pos.device,
                    dtype=tet_pos.dtype,
                )

        # --------------------------------------------------
        # Optional return: offsets only
        # --------------------------------------------------
        if return_offset:
            return ori_pos_delta

        # --------------------------------------------------
        # Inference returns
        # --------------------------------------------------
        if inference:
            if return_surf:
                return (
                    amips_energy,
                    edge,
                    area_variance,
                    surface_align,
                    normal_loss,
                    occ_loss,
                    occ_iou,
                    lap,
                    delta_loss,
                    tet_pos,
                    all_pred_occ_prob,
                    condition,
                    surface,
                    pred_surface,
                    other_chamfer_distance,
                    sum_time,
                )

            return (
                amips_energy,
                edge,
                area_variance,
                surface_align,
                normal_loss,
                occ_loss,
                occ_iou,
                lap,
                delta_loss,
                tet_pos,
                all_pred_occ_prob,
                condition,
                other_chamfer_distance,
            )

        # --------------------------------------------------
        # Training returns
        # --------------------------------------------------
        if not return_all:
            return (
                amips_energy,
                edge,
                area_variance,
                surface_align,
                normal_loss,
                occ_loss,
                lap,
                delta_loss,
                other_chamfer_distance,
                lap_v_loss,
            )

        return (
            amips_energy,
            edge,
            area_variance,
            surface_align,
            normal_loss,
            occ_loss,
            lap,
            delta_loss,
            tet_pos,
            z,
            encoding_occ,
            pred_occ_prob,
            gt_occ,
            other_chamfer_distance,
            None,
            lap_v_loss,
        )