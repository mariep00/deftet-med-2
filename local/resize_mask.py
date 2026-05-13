import numpy as np
import nibabel as nib
from scipy import ndimage


def load_mask(input_file):
    """
    Load a NIfTI mask file.

    Returns:
        img: original NIfTI image
        mask: boolean mask
        voxel_size: voxel size in mm
    """
    img = nib.load(input_file)
    data = img.get_fdata()
    mask = data > 0
    voxel_size = img.header.get_zooms()[:3]

    return img, mask, voxel_size


def mask_info(input_file):
    """
    Print basic information about the mask.
    """
    img, mask, voxel_size = load_mask(input_file)

    coords = np.argwhere(mask)

    voxel_count = int(mask.sum())
    voxel_volume = float(np.prod(voxel_size))
    volume_mm3 = voxel_count * voxel_volume

    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_size_vox = bbox_max - bbox_min + 1
    bbox_size_mm = bbox_size_vox * np.array(voxel_size)

    center_vox = coords.mean(axis=0)
    center_world = nib.affines.apply_affine(img.affine, center_vox)

    approx_radius_mm = bbox_size_mm.max() / 2

    info = {
        "shape": img.shape,
        "voxel_size_mm": voxel_size,
        "voxel_count": voxel_count,
        "volume_mm3": volume_mm3,
        "bounding_box_min_vox": bbox_min,
        "bounding_box_max_vox": bbox_max,
        "bounding_box_size_mm": bbox_size_mm,
        "center_vox": center_vox,
        "center_world_mm": center_world,
        "approx_radius_mm": approx_radius_mm,
        "approx_diameter_mm": approx_radius_mm * 2,
    }

    for key, value in info.items():
        print(f"{key}: {value}")

    return info


def make_spherical_kernel(radius_mm, voxel_size):
    """
    Create a spherical structuring element.

    Args:
        radius_mm: radius of the kernel in millimeters
        voxel_size: voxel size from the NIfTI header
    """
    radius_vox = np.ceil(radius_mm / np.array(voxel_size)).astype(int)

    x = np.arange(-radius_vox[0], radius_vox[0] + 1)
    y = np.arange(-radius_vox[1], radius_vox[1] + 1)
    z = np.arange(-radius_vox[2], radius_vox[2] + 1)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    distances_mm = np.sqrt(
        (X * voxel_size[0]) ** 2 +
        (Y * voxel_size[1]) ** 2 +
        (Z * voxel_size[2]) ** 2
    )

    return distances_mm <= radius_mm


def resize_mask(input_file, output_file, change_mm):
    """
    Grow or shrink a binary mask.

    Args:
        input_file: path to input NIfTI mask
        output_file: path to output NIfTI mask
        change_mm: positive grows mask, negative shrinks mask

    Example:
        resize_mask("T1_bin.nii", "T1_bin_bigger_5mm.nii", 5)
        resize_mask("T1_bin.nii", "T1_bin_smaller_5mm.nii", -5)
    """
    img, mask, voxel_size = load_mask(input_file)

    radius_mm = abs(change_mm)

    if radius_mm == 0:
        resized = mask
    else:
        kernel = make_spherical_kernel(radius_mm, voxel_size)

        if change_mm > 0:
            resized = ndimage.binary_dilation(mask, structure=kernel)
        else:
            resized = ndimage.binary_erosion(mask, structure=kernel)

    resized = resized.astype(np.uint8)

    new_img = nib.Nifti1Image(resized, img.affine, img.header)
    new_img.set_data_dtype(np.uint8)

    nib.save(new_img, output_file)

    return resized


def save_mask(mask, reference_img, output_file):
    """
    Save a mask using the affine and header from a reference image.
    """
    mask = mask.astype(np.uint8)

    new_img = nib.Nifti1Image(mask, reference_img.affine, reference_img.header)
    new_img.set_data_dtype(np.uint8)

    nib.save(new_img, output_file)


import numpy as np
import nibabel as nib


def create_cube_mask_from_reference(reference_file, output_file, center_vox, size_vox):
    """
    Create a cube mask using voxel coordinates.

    Args:
        reference_file: NIfTI file to copy shape, affine, and header from
        output_file: output cube mask NIfTI file
        center_vox: cube center in voxel coordinates, for example [128, 128, 128]
        size_vox: cube size in voxels. Can be one number or [sx, sy, sz]

    Example:
        create_cube_mask_from_reference(
            "T1_bin.nii",
            "cube_mask_40vox.nii",
            center_vox=[128, 128, 128],
            size_vox=40
        )
    """
    ref_img = nib.load(reference_file)
    shape = ref_img.shape[:3]

    center_vox = np.array(center_vox, dtype=int)

    if np.isscalar(size_vox):
        size_vox = np.array([size_vox, size_vox, size_vox], dtype=int)
    else:
        size_vox = np.array(size_vox, dtype=int)

    half_size = size_vox // 2

    start = center_vox - half_size
    end = start + size_vox

    # Keep cube inside image bounds
    start = np.maximum(start, 0)
    end = np.minimum(end, shape)

    mask = np.zeros(shape, dtype=np.uint8)

    mask[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ] = 1

    cube_img = nib.Nifti1Image(mask, ref_img.affine, ref_img.header)
    cube_img.set_data_dtype(np.uint8)

    nib.save(cube_img, output_file)

    return mask

import numpy as np
import nibabel as nib
from scipy import ndimage


def resize_cube_mask(input_file, output_file, change_vox):
    """
    Grow or shrink a cube mask while keeping sharp cube corners.

    Args:
        input_file: input cube mask NIfTI file
        output_file: output resized cube mask NIfTI file
        change_vox: positive grows cube, negative shrinks cube, in voxels

    Example:
        resize_cube_mask("cube_mask_40vox.nii", "cube_bigger_10vox.nii", 10)
        resize_cube_mask("cube_mask_40vox.nii", "cube_smaller_10vox.nii", -10)
    """
    img = nib.load(input_file)
    data = img.get_fdata()

    mask = data > 0

    n = abs(int(change_vox))

    if n == 0:
        resized = mask
    else:
        kernel_size = 2 * n + 1
        box_kernel = np.ones(
            (kernel_size, kernel_size, kernel_size),
            dtype=bool
        )

        if change_vox > 0:
            resized = ndimage.binary_dilation(mask, structure=box_kernel)
        else:
            resized = ndimage.binary_erosion(mask, structure=box_kernel)

    resized = resized.astype(np.uint8)

    new_img = nib.Nifti1Image(resized, img.affine, img.header)
    new_img.set_data_dtype(np.uint8)

    nib.save(new_img, output_file)

    return resized