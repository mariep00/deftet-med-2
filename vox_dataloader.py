'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
'''
Voxel Mask -> Surface Point Cloud + SDF DataLoader

1. Loads binary MRI voxel masks (.nii or .nii.gz)
2. Samples surface point clouds directly from the mask
3. Samples random 3D points and computes SDF values
4. Returns everything in a batched dictionary

The code is intentionally kept close to the original mesh-based ShapeNet loader.
The main difference is that the input is a voxel mask instead of an input mesh.
'''

#import kaolin as kal
#from matplotlib import scale
import torch
import os
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import binary_erosion, generate_binary_structure

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", DEVICE)

#TODO use this function to convert the voxel coordinates to the normalized coordinates that are used for the model input check if correct and if necessary to use this in the dataloader or in the training loop or somewhere else
def voxel_zyx_to_normalized_xyz(points_zyx, mask_shape, spacing):
    """
    Convert voxel coordinates from z,y,x index space to normalized x,y,z model space.

    Args:
        points_zyx:
            ndarray [N, 3]
            Coordinates in voxel index order z,y,x.

        mask_shape:
            tuple (D, H, W)

        spacing:
            ndarray [3]
            Voxel spacing from the NIfTI header.

    Returns:
        points_xyz:
            ndarray [N, 3]
            Coordinates in normalized x,y,z model space.
            Values are approximately inside [-0.5, 0.5].
    """

    points_zyx = np.asarray(points_zyx, dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)

    # Convert z,y,x voxel coordinates to x,y,z physical coordinates.
    points_xyz = points_zyx[:, ::-1] * spacing[::-1]

    # Convert D,H,W volume shape to W,H,D physical size in x,y,z order.
    physical_size_xyz = np.asarray(mask_shape[::-1], dtype=np.float32) * spacing[::-1]

    center_xyz = physical_size_xyz / 2.0
    scale = np.max(physical_size_xyz)

    points_xyz = (points_xyz - center_xyz) / scale

    return points_xyz.astype(np.float32)



#TODO: chnage this class to remve the unnnecessary stuff like canonical 
class MRIVoxelDataset(Dataset):
    """
    Dataset for MRI voxel masks stored as .nii or .nii.gz files.

    Args:
        source (str): Root folder containing MRI mask files.
        train (bool): Compatibility argument kept close to the original loader.
        file_list (list[str] or None): Optional explicit list of file paths.

    Returns:
        tuple:
            voxel_dict (dict):
                mask: FloatTensor [1, D, H, W]
                spacing: FloatTensor [3], for example the z-direction has thicker slices
                affine: FloatTensor [4, 4], if want to map predictions back into real world coordinates
            attributes (dict):
                name: str
                synset: str
    """

    def __init__(self, source, train=True, file_list=None):
        self.source = source
        self.train = train

        if file_list is not None:
            self.paths = list(file_list)
        else:
            self.paths = []
            for root, _, files in os.walk(source):
                for file_name in files:
                    if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                        self.paths.append(os.path.join(root, file_name))
            self.paths = sorted(self.paths)

        self.names = [self._path_to_name(p) for p in self.paths]
        self.synset_idxs = ['mri' for _ in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if nib is None:
            raise RuntimeError('nibabel is required to read MRI NIfTI files')

        nii = nib.load(self.paths[idx])

        mask = np.asarray(nii.get_fdata(dtype=np.float32), dtype=np.float32)
        if mask.ndim != 3:
            raise ValueError(f'Expected a 3D volume, got shape {mask.shape}')

        mask = (mask > 0).astype(np.float32)
        if mask.sum() == 0:
            raise ValueError(f'Empty foreground mask in file: {self.paths[idx]}') # TODO 0 is foreeground, 1 is background?

        voxel_dict = {
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'spacing': torch.tensor(np.asarray(nii.header.get_zooms()[:3], dtype=np.float32)),
            'affine': torch.tensor(np.asarray(nii.affine, dtype=np.float32)),
        }
        attributes = { # TODO attributes
            'name': self.names[idx],
            'synset': self.synset_idxs[idx],
        }
        return voxel_dict, attributes

    @staticmethod
    def _path_to_name(path):
        base = os.path.basename(path)
        if base.endswith('.nii.gz'):
            return base[:-7]
        if base.endswith('.nii'):
            return base[:-4]
        return os.path.splitext(base)[0]


# ------------------------------------------------------------
# Alternative Step 1: Convert a binary mask -> points point cloud
# MRI mask
# -> surface point cloud directly from mask
# -> SDF samples directly from mask
# -> model input and supervision
# ------------------------------------------------------------
class SamplePointsFromMask:
    """
    Sample a fixed number of points directly from a binary mask.

    This produces a hollow point cloud containing only surface samples.
    """

    def __init__(self, num_points, save_preprocess=False):
        self.num_points = num_points
        self.save_preprocess = save_preprocess

    def __call__(self, voxel_data):
        """
        Args:
            voxel_data (dict):
                mask: Tensor [1, D, H, W]
                spacing: Tensor [3]

        Returns:
            points: Tensor [N, 3], normalized xyz coordinates
        """
        num_points = self.num_points

        mask = voxel_data['mask'].squeeze(0).cpu().numpy().astype(np.uint8) 
        spacing = voxel_data['spacing'].cpu().numpy().astype(np.float32)

        mask = mask > 0

        se = generate_binary_structure(3, 3)
        eroded = binary_erosion(mask, structure=se, iterations=1)
        surface = mask ^ eroded

        coords_zyx = np.argwhere(surface)

        if coords_zyx.shape[0] == 0:
            raise ValueError("Mask has no surface voxels. Check if mask is empty or fully filled.")

        if coords_zyx.shape[0] >= num_points:
            idx = np.random.choice(coords_zyx.shape[0], num_points, replace=False)
        else:
            idx = np.random.choice(coords_zyx.shape[0], num_points, replace=True) # If not enough surface points, allow duplicates

        points_zyx = coords_zyx[idx].astype(np.float32)

        # TODO: check if this fixed the issue with the order of the coordinates and if the points are correctly normalized to the range [-0.5, 0.5] in x,y,z order
        points_xyz = voxel_zyx_to_normalized_xyz(
            points_zyx,
            mask_shape=mask.shape,
            spacing=spacing,
        )

        return torch.from_numpy(points_xyz.copy()).cpu()


    def __repr__(self):
        if not self.save_preprocess:
            return 'point_cloud_%s' % (str(datetime.now()))
        return 'point_cloud'

# TODO understand how this changes from the mesh to sdf this is the equivalent from kaolin_mesh_to_sdf but need to understand how it chnages if lost precisipon or not 
def voxel_to_sdf(mask, spacing):
    """
    Compute an approximate signed distance field from a binary voxel mask.

    positive = inside foreground
    negative = outside foreground
    
    Args:
        mask (ndarray): Binary array [D, H, W]
        spacing (ndarray): Physical voxel spacing [3]

    Returns:
        sdf (ndarray): Signed distance field [D, H, W]
    """
    if ndimage is None:
        raise RuntimeError('scipy is required for SDF computation from voxel masks')
    
    mask = mask.astype(bool)
    spacing = np.asarray(spacing, dtype=np.float32)
    
    # TODO: check if correct positive inside and negative outside (try printing)
    inside_dt = ndimage.distance_transform_edt(mask, sampling=spacing)
    outside_dt = ndimage.distance_transform_edt(~mask, sampling=spacing) #TODO check if this is correct and if the sign is correct (try printing)
    sdf = inside_dt - outside_dt

    # Same scale used to normalize xyz points
    physical_size = np.array(mask.shape[::-1], dtype=np.float32) * spacing[::-1]
    scale = np.max(physical_size)

    sdf = sdf / scale

    return sdf


class SDFPoints:
    """
    Sample random 3D points and compute SDF values.

    These points fill space and are used as supervision.
    """
#TODO chaeck if necessary to have the normals 
    def __init__(self, num_points, save_preprocess=False):
        self.num_points = num_points
        self.save_preprocess = save_preprocess

    def __call__(self, voxel_data):
        """
        Args:
            voxel_data (dict):
                mask: Tensor [1, D, H, W]
                spacing: Tensor [3]

        Returns:
            tuple:
                points: Tensor [N, 3], normalized xyz coordinates
                sdf: Tensor [N]
        """
        mask = voxel_data['mask'].squeeze(0).cpu().numpy().astype(np.uint8)
        spacing = voxel_data['spacing'].cpu().numpy().astype(np.float32)

        sdf_grid = voxel_to_sdf(mask, spacing)

        d, h, w = mask.shape

        pts_zyx = np.stack([
            np.random.uniform(0, d - 1, size=self.num_points),
            np.random.uniform(0, h - 1, size=self.num_points),
            np.random.uniform(0, w - 1, size=self.num_points),
        ], axis=1).astype(np.float32)

        sdf_vals = ndimage.map_coordinates(
            sdf_grid,
            [pts_zyx[:, 0], pts_zyx[:, 1], pts_zyx[:, 2]],
            order=1,
            mode='nearest'
        ).astype(np.float32)

        
        points_xyz = voxel_zyx_to_normalized_xyz(
            pts_zyx,
            mask_shape=mask.shape,
            spacing=spacing,
        )

        return torch.from_numpy(points_xyz.copy()), torch.from_numpy(sdf_vals.copy())

    def __repr__(self):
        if not self.save_preprocess:
            return 'sdf_%s' % (str(datetime.now()))
        return 'sdf'


# TODO what transforms are applied now ? 
class ProcessedMRIDataset(Dataset):
    """
    Apply a preprocessing transform to a base MRI dataset.

    Args:
        dataset (Dataset): Base dataset.
        transform (callable): Transform applied to each sample.
        num_workers (int): Compatibility argument kept close to the original code.
        cache_dir (str): Optional directory for saving processed outputs.
    """

    def __init__(self, dataset, transform, num_workers=0, cache_dir=None):
        self.dataset = dataset
        self.transform = transform
        self.num_workers = num_workers
        self.cache_dir = cache_dir

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        base_data, attrs = self.dataset[idx]
        processed = self.transform(base_data)
        return processed, attrs


#TODO keep for now but might want to remove 
class CombinationDataset(Dataset):
    """
    Combine several datasets sharing the same indexing.

    Args:
        datasets (list[Dataset]): Datasets with equal length.
    """

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        processed_items = []
        attrs = None
        for dataset in self.datasets:
            item, item_attrs = dataset[idx]
            processed_items.append(item)
            if attrs is None:
                attrs = item_attrs
        return processed_items, attrs



#TODO: is the cache really saved here and used ? am i missing any step? 
def create_dataloader(mri_source='/work3/s233736/datasets/MRI',
                      save_cache_root='/work3/s233736/deftet_runs/run_01',
                      train=True, batch_size=1, add_occupancy=False, only_chairs=False): #TODO check for add occupancy and only chairs def create_dataloader(shapenet_source='/work3/s233736/datasets/shapenetcore', #TODO: changeD to have just one 03001627 save_cache_root = '/work3/s233736/deftet_runs/run_01', train=True, batch_size=1, add_occupancy=False, only_chairs=False)
    """
    Create full dataloader pipeline.

    Args:
        shapenet_source (str): Root directory containing binary MRI voxel masks.
        save_cache_root (str): Root directory for cached processed outputs.
        train (bool): Compatibility argument kept close to the original code.
        batch_size (int): Batch size.
        add_occupancy (bool): Compatibility argument kept close to the original code.

   Returns:
    DataLoader yielding batches containing:
        sample_points: Tensor [B, N_surface, 3]
        name: list[str]
        synset: list[str]
        sdf_point: Tensor [B, N_sdf, 3]
        sdf_value: Tensor [B, N_sdf]
        spacing: Tensor [B, 3]
        affine: Tensor [B, 4, 4]
    """

    # ds = MRIVoxelDataset(source=shapenet_source, train=train)
    print("create_dataloader source:", mri_source)
    ds = MRIVoxelDataset(source=mri_source, train=train)
    print("dataset length:", len(ds))
    if len(ds) > 0:
        print("first file:", ds.paths[0])

    sv_dir = os.path.join(save_cache_root, 'surface_points') # TODO do i really need this ? 
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    
    # TODO: 1. surface points 
    processed_ds = ProcessedMRIDataset(
        ds, # TODO: here u chnaged frim watertight_mesh to ds because i want to sample points from the mask directly instead of the mesh check if correct 
        SamplePointsFromMask(100000, save_preprocess=True),
        num_workers=0,
        cache_dir=sv_dir)

    # TODO: 2. sdf points
    print('==> preprocess sdf')
    sv_dir = os.path.join(save_cache_root, 'sdf') # TODO do i really need this ? i noticed that i dont need the watertightness so what should i do about this ?
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    occ_dataset = ProcessedMRIDataset(
        ds,
        SDFPoints(100000, save_preprocess=True),
        num_workers=0,
        cache_dir=sv_dir)

    combined_dataset = CombinationDataset([ds, processed_ds, occ_dataset]) # TODO why do i have thsi ? and is it necessary now that i dont use the water toght mesh ? 



 # TODO ? 
    def collate_fn(batch_list):
        """Custom batching for combined MRI outputs.""" # TODO why? 
        data = dict()

        #data['verts'] = [da[0][0][0] for da in batch_list]
        #data['faces'] = [da[0][0][1] for da in batch_list]
        data['sample_points'] = torch.cat([da[0][1].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['name'] = [da[1]['name'] for da in batch_list]
        data['synset'] = [da[1]['synset'] for da in batch_list]
        data['sdf_point'] = torch.cat([da[0][2][0].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['sdf_value'] = torch.cat([da[0][2][1].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['spacing'] = torch.stack([da[0][0]['spacing'] for da in batch_list], dim=0)
        data['affine'] = torch.stack([da[0][0]['affine'] for da in batch_list], dim=0)
        return data

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False, #TODO check if i should shuffle or not
        num_workers=0, # TODO chnaged from 8 to 0 why ? 
        collate_fn=collate_fn,
        drop_last=True,
    )
    return dataloader


if __name__ == '__main__':
    """Debug: inspect MRI point and SDF batches."""    
    dataloader = create_dataloader(train=False) #TODO check if i should do the only chairs or not
    print('==> finished validation data')

    #TODO FOR NOW COMMENTED OUT 
    # save_folder = '/work3/s233736/deftet_runs/run_01/npz_samples' # TODO check if i really need this and if i do where should i save the meshes ?
    #os.makedirs(save_folder, exist_ok=True)

    from tqdm import tqdm
    #cnt = 0
    for data in tqdm(iter(dataloader)): # TODO WHAT ARE THESE BASHES 

        '''
        # TODO: for later when i want to save the data 
        sample_points_batch = data['sample_points']
        sdf_point_batch = data['sdf_point']
        sdf_value_batch = data['sdf_value']
        name_list = data['name']

        for sample_points, sdf_points, sdf_values, name in zip(
            sample_points_batch,
            sdf_point_batch,
            sdf_value_batch,
            name_list
        ):
            out_path = os.path.join(save_folder, name + '.npz')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            np.savez(
                out_path,
                sample_points=sample_points.cpu().numpy(),
                sdf_point=sdf_points.cpu().numpy(),
                sdf_value=sdf_values.cpu().numpy(),
            )

            cnt += 1

            if cnt > 100:
                exit()'''
        
        #name_list = data['name']
        print('sample_points:', data['sample_points'].shape) # [B, N_surface, 3]
        print('sdf_point:', data['sdf_point'].shape) # [B, N_sdf, 3]
        print('sdf_value:', data['sdf_value'].shape) # [B, N_sdf]
        print('name:', data['name']) # list[str]

