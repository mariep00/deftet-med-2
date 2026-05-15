'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
'''
Mesh -> Watertight Mesh -> Point Cloud + SDF DataLoader

This module builds a PyTorch DataLoader that:
1. Loads ShapeNet meshes (.obj)
2. Converts them into watertight meshes via voxelization
3. Samples surface point clouds (hollow)
4. Samples random 3D points and computes SDF values
5. Returns everything in a batched dictionary
'''
import kaolin as kal
import torch
import os
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import meshio
from types import SimpleNamespace
from torch.utils.data import Dataset

class MSHDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.paths = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".msh")
        ])
        self.names = [
            os.path.splitext(os.path.basename(p))[0]
            for p in self.paths
        ]
        self.synset_idxs = ["msh" for _ in self.paths]

    def __len__(self):
        return len(self.paths)

    def get_data(self, idx):
        path = self.paths[idx]
        msh = meshio.read(path)

        vertices = torch.tensor(msh.points[:, :3], dtype=torch.float32)

        if "triangle" not in msh.cells_dict:
            raise ValueError(
                f"{path} has no triangle surface cells. "
                f"Available cell types: {list(msh.cells_dict.keys())}"
            )

        faces = torch.tensor(msh.cells_dict["triangle"], dtype=torch.long)

        return SimpleNamespace(vertices=vertices, faces=faces)

    def get_attributes(self, idx):
        return {
            "name": self.names[idx],
            "synset": "msh",
            "path": self.paths[idx],
        }

    def get_cache_key(self, idx):
        return self.names[idx]

    def __getitem__(self, idx):
        return kal.io.dataset.KaolinDatasetItem(
            self.get_data(idx),
            self.get_attributes(idx),
        )


# TODO from utils.mesh_utils import save_mesh

# ------------------------------------------------------------
# Step 1: Convert raw mesh -> watertight mesh
# ------------------------------------------------------------
class MakeSurfaceMesh:
    """
    Convert a raw mesh into a watertight mesh.

    mesh (vertices, faces)
        -> normalize + center
        -> voxel grid (discretization)
        -> ODM projection (fills holes)
        -> mesh extraction
        -> smoothing
        -> rescale back

    Output:
        vertices: Tensor [V, 3]
        faces: Tensor [F, 3]
    """
    def __init__(self, resolution=64, smoothing_iterations=2, save_preprocess=False, max_length=0.9): #bef: resolution=100, smoothing_iterations=3
        """
        Args:
            resolution (int): voxel grid resolution
            smoothing_iterations (int): Laplacian smoothing steps
            save_preprocess (bool): caching flag
            max_length (float): normalization scale
        """
        self.resolution = resolution
        self.smoothing_iterations = smoothing_iterations
        self.save_preprocess = save_preprocess
        self.max_length = max_length
        self.error_idx = []

    def __call__(self, mesh):
        """
        Args:
            mesh: Kaolin mesh object with attributes:
                - vertices [V, 3]
                - faces [F, 3]

        Returns:
            (new_vertices, new_faces): watertight mesh
        """

        # Extract original mesh geometry
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()

        # Normalize scale
        max_l = max(vertices[..., 0].max() - vertices[..., 0].min(),
                    vertices[..., 1].max() - vertices[..., 1].min(),
                    vertices[..., 2].max() - vertices[..., 2].min())
        vertices = (vertices / max_l) * self.max_length

        # Center mesh
        mid_p = (vertices.max(dim=0)[0] + vertices.min(dim=0)[0]) / 2
        vertices = vertices - mid_p.unsqueeze(dim=0)

        # Mesh -> voxel grid
        voxelgrid = kal.ops.conversions.trianglemeshes_to_voxelgrids(
                vertices.unsqueeze(0), faces,
                resolution=self.resolution)
        
        # Fill holes using orthographic depth maps (watertight conversion)
        odms = kal.ops.voxelgrid.extract_odms(voxelgrid)
        voxelgrid = kal.ops.voxelgrid.project_odms(odms)

       # Voxel grid -> mesh
        new_vertices, new_faces = kal.ops.conversions.voxelgrids_to_trianglemeshes(
            voxelgrid,
        )
        new_vertices = new_vertices[0]
        new_faces = new_faces[0]

        # laplacian smoothing
        adj_mat = kal.ops.mesh.adjacency_matrix(
            new_vertices.shape[0],
            new_faces)

        num_neighbors = torch.sparse.sum(
            adj_mat, dim=1).to_dense().view(-1, 1)

        for i in range(self.smoothing_iterations):
            neighbor_sum = torch.sparse.mm(adj_mat, new_vertices)
            new_vertices = neighbor_sum / num_neighbors


        # normalize / rescale to original 
        orig_min = vertices.min(dim=0)[0]
        orig_max = vertices.max(dim=0)[0]
        new_min = new_vertices.min(dim=0)[0]
        new_max = new_vertices.max(dim=0)[0]

        new_vertices = (new_vertices - new_min) / (new_max - new_min)
        new_vertices = new_vertices * (orig_max - orig_min) + orig_min

        return new_vertices.cpu(), new_faces.cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'watertight_%s'%(str(datetime.now()))
        return 'watertight'

class SamplePointsFromMesh:
    """
    Sample a fixed number of points from the mesh surface. This produces a HOLLOW point cloud (surface only).
    """

    def __init__(self, num_points, with_normals=True, save_preprocess=False):
        self.num_points = num_points
        self.with_normals = with_normals
        self.save_preprocess = save_preprocess

    def __call__(self, mesh):
        """
        Args:
            mesh: (vertices, faces)

        Returns:
            points: [N, 3]
            normals (optional): [N, 3]
        """

        vertices = mesh[0].unsqueeze(dim=0).float().cuda()
        faces = mesh[1].long().cuda()

        # Sample points ON triangles
        points, face_choices = kal.ops.mesh.sample_points(
            vertices, faces, self.num_points)

        if self.with_normals:
            face_vertices = kal.ops.mesh.index_vertices_by_faces(vertices, faces)
            face_normals = kal.ops.mesh.face_normals(
                face_vertices, unit=True)
            normals = face_normals[face_choices]
            return points.squeeze(0), normals.squeeze(0)

        return points.squeeze(0).cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'point_cloud_%s'%(str(datetime.now()))
        return 'point_cloud'

def kaolin_mesh_to_sdf(verts_bxnx3, face_fx3, points_bxnx3):
    """
    Compute signed distance from points to mesh.

    Args:
        verts_bxnx3: [B, V, 3]
        face_fx3: [F, 3]
        points_bxnx3: [B, N, 3]

    Returns:
        sdf: [B, N]
    """
    sign = kal.ops.mesh.check_sign(verts_bxnx3, face_fx3, points_bxnx3, hash_resolution=512)
    face_vertices = kal.ops.mesh.index_vertices_by_faces(verts_bxnx3, face_fx3)
    distance, index, dist_type = kal.metrics.trianglemesh.point_to_mesh_distance(points_bxnx3, face_vertices)
    sign = sign.float() * 2.0 - 1.0  # (1: inside; -1: outside)
    sdf = sign * distance
    return sdf


class SDFPoints:
    """
    Sample random 3D points and compute SDF values.

    These points fill space (not hollow) and are used as supervision.
    """

    def __init__(self, num_points, with_normals=True, save_preprocess=False):
        self.num_points = num_points
        self.with_normals = with_normals
        self.save_preprocess = save_preprocess

    def __call__(self, mesh):
        vertices = mesh[0].unsqueeze(dim=0).float().cuda()
        faces = mesh[1].long().cuda()

        # Uniform random points in cube
        points = 1.05 * (torch.rand(1, self.num_points, 3).cuda() - .5)

        # Compute SDF
        sdf = kaolin_mesh_to_sdf(vertices, faces, points)

        return points[0].cpu(), sdf[0].cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'sdf_%s'%(str(datetime.now()))
        return 'sdf'


def create_dataloader(msh_source='/work3/s233736/datasets/mesh_surfaces',
                      save_cache_root = '/work3/s233736/deftet_runs/run_01',
                      train=True, batch_size=1, add_occupancy=False, only_chairs=False,
                      val_count=2): # Bef: train=True, batch_size=8, only_chairs=False
    """
        Create full dataloader pipeline.

        Returns batches containing:
            verts: list of meshes
            faces: list of meshes
            sample_points: [B, N, 3]
            sdf_point: [B, N, 3]
            sdf_value: [B, N]
    """

    train_cat = [   '02691156',
                         '02828884',
                         '02933112',
                         '02958343',
                         '03001627',
                         '03211117',
                         '03636649',
                         '03691459',
                         '04090263',
                         '04256520',
                         '04379243',
                         '04401088',
                         '04530566']
    if only_chairs:
        train_cat = [
            '03001627',
        ]

    # train_cat = ['02958343'] # car shape##########
    # ds = kal.io.shapenet.ShapeNetV1(root=shapenet_source, categories=train_cat,
                                    #with_materials=False, train=train)
    ds = MSHDataset(msh_source)

    # Remove broken model
    error_model = ['04090263_4a32519f44dc84aabafe26e2eb69ebf4'] # This one has no mesh :(
    error_idx = [ds.names.index(e) for e in error_model if e in ds.names]
    for idx in error_idx:
        ds.paths.pop(idx)
        ds.synset_idxs.pop(idx)
        ds.names.pop(idx)

    if val_count < 0:
        raise ValueError(f'val_count must be non-negative, got {val_count}')
    if val_count >= len(ds):
        raise ValueError(
            f'val_count={val_count} leaves no training meshes for dataset of size {len(ds)}'
        )

    if val_count > 0:
        split_slice = slice(None, -val_count) if train else slice(-val_count, None)
    else:
        split_slice = slice(None)

    ds.paths = ds.paths[split_slice]
    ds.synset_idxs = ds.synset_idxs[split_slice]
    ds.names = ds.names[split_slice]

    split_name = 'train' if train else 'val'
    print(f'==> Using {split_name} split with {len(ds.names)} meshes:')
    print(ds.names)

    sv_dir = os.path.join(save_cache_root, 'watertight')
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    print('==> preprocess watertight mesh')

     # 1: watertight mesh
    watertight_mesh = kal.io.dataset.ProcessedDataset(
        ds, MakeSurfaceMesh(100, 3, save_preprocess=True), num_workers=0,
        cache_dir=sv_dir)


    sv_dir = os.path.join(save_cache_root, 'pcd')
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    print('==> preprocess point cloud')
    ####################

    # 2: surface points
    processed_ds = kal.io.dataset.ProcessedDataset(
        watertight_mesh, SamplePointsFromMesh(100000, with_normals=False, save_preprocess=True),
        num_workers=0,
        cache_dir=sv_dir)

    print('==> preprocess sdf')
    sv_dir = os.path.join(save_cache_root, 'sdf')
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)


    # 3: SDF samples
    occ_dataset = kal.io.dataset.ProcessedDataset(
        watertight_mesh, SDFPoints(100000, save_preprocess=True),
        num_workers=0,
        cache_dir=sv_dir)
    #########


    combined_dataset = kal.io.dataset.CombinationDataset([watertight_mesh, processed_ds,
                                                          occ_dataset])

    def collate_fn(batch_list):
        """Custom batching because meshes have variable size"""
        data = dict()

        data['verts'] = [da[0][0][0] for da in batch_list]
        data['faces'] = [da[0][0][1] for da in batch_list]
        data['sample_points'] = torch.cat([da[0][1].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['name'] = [da[1][0]['name'] for da in batch_list]
        data['synset'] = [da[1][0]['synset'] for da in batch_list]
        data['sdf_point'] = torch.cat([da[0][2][0].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['sdf_value'] = torch.cat([da[0][2][1].unsqueeze(dim=0) for da in batch_list], dim=0)
        return data

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=train,
    )##### We always shuffle the data here
    return dataloader

if __name__ == '__main__':
    from utils.mesh_utils import save_mesh #TODO
    """Debug: save processed meshes"""
    # dataloader = create_dataloader(train=False, only_chairs=False)
    dataloader = create_dataloader(train=False, only_chairs=False) # Bef: only_chairs=False
    print('==> finished validatation data')
    # dataloader_val = create_dataloader(train=False, only_chairs=True)
    save_folder = '/work3/s233736/deftet_runs/run_01/objs' #Bef: '/root/shapenet_car_all_update'
    os.makedirs(save_folder, exist_ok=True)
    from tqdm import tqdm
    cnt = 0
    for data in tqdm(iter(dataloader)):
        # import ipdb
        # ipdb.set_trace()######
        mesh_v_list = data['verts']
        mesh_f_list = data['faces']
        name_list = data['name']
        for v, f, n in zip(mesh_v_list, mesh_f_list, name_list):
            # import ipdb
            # ipdb.set_trace()
            # just before save_mesh(...)
            out_path = os.path.join(save_folder, n + '.obj')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_mesh(v.data.cpu().numpy(), f.data.cpu().numpy(), out_path)

            #save_mesh(v.data.cpu().numpy(), f.data.cpu().numpy(), os.path.join(save_folder, n  + '.obj'))
            cnt += 1
            if cnt > 100:
                exit()




