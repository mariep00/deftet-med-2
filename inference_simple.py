'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

# Minimal inference-only version.
# Goal: keep only what is necessary to load one trained model, run one sample,
# extract the predicted mesh and optionally save the result.


from config import OPTIONS
from simple_parallel import SimpleParallelWrapper
from utils.experiment import Experiment
from utils import tet_utils
from utils.mesh_utils import save_mesh
import argparse
import utils.dataloder_helper as helpers
from layers.DefTet.deftet import DefTet
import numpy as np
import os
import torch
import torch.nn as nn
from utils.matrix_utils import MySparse
from layers.pc_model import DeformableTetNetwork
from dataloader import create_dataloader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FOLDER_PATH = os.path.join(ROOT_DIR, 'experiments')

# Ensure reproducibility
np.random.seed(1)
torch.random.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3)


# TODO keep like this but change the help so it is accurate to what the parser actually gives.
# Resolved: updated help texts, removed the stray `1`.
# Also removed flags that only existed for metrics / alternate save branches.

# ==============================
# Argument parsing
# ==============================
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path to experiment to load')
    parser.add_argument('--step', type=int,
                        help='Checkpoint step to load, use 0 for default best checkpoint', default=0)
    parser.add_argument('--save', action='store_true',
                        help='Save predicted tet grid and surface mesh to an NPZ file', default=False)
    return parser.parse_args()


# ==============================
# Engine class
# ==============================
class Engine(object):
    """
    Minimal inference engine:
    - Initialises tetrahedral mesh
    - Builds model and deformation module
    - Runs inference
    - Saves prediction if requested

    Removed from the original role:
    - metric computation
    - training dataloader support
    - timing mode
    - smooth/fix branches
    - extra evaluation bookkeeping
    """

    # TODO understand if this is all needed in my case or if I can remove some of the arguments
    # Resolved: most of them were not needed for minimal inference.
    # Kept only config, validation dataloader, and save flag.
    def __init__(self,
                 config=None,
                 dataloader_val=None,
                 save=False):

        # Configuration and runtime flags
        self.save = save
        self.config = config
        self.dataloader_val = dataloader_val
        self.deftet = DefTet()

        # ----------------------------------
        # Initialize tetrahedral mesh
        # ----------------------------------
        vertices_nx3, tetrahedron_fx4, mask = helpers.read_tetrahedron(
            res=self.config.res, root=ROOT_DIR)

        # Vertex positions (centered at origin)
        self.init_tet_pos = torch.from_numpy(vertices_nx3).float().to(
            self.config.device) - 0.5

        # Connectivity mask / validity mask
        self.init_pos_mask = torch.from_numpy(
            mask).float().to(self.config.device)

        # Tetrahedra defined by 4 vertex indices
        self.init_tet_fx4 = torch.from_numpy(
            tetrahedron_fx4).long().to(self.config.device)

        # ----------------------------------
        # Build adjacency structure
        # ----------------------------------
        # Used for graph-based operations on tetrahedral vertices.
        # This is still needed because the model expects it.
        self.point_adj_sparse = tet_utils.c_tet_to_adj_sparse(
            vertices_nx3, tetrahedron_fx4, normalize=True).to(self.config.device)

        self.point_adj_sparse = MySparse(self.point_adj_sparse)

        # ----------------------------------
        # Precompute face information
        # ----------------------------------
        # Converts tetrahedra into triangular faces.
        # Still needed because ParallelWrapper expects tet-face connectivity.
        tet_face_fx3, tet_facetet_idx_fx2, _, _ = tet_utils.tet_to_face(
            vertices_nx3.shape[0],
            tetrahedron_fx4
        )

        self.tet_face_fx3 = torch.from_numpy(tet_face_fx3).long().to(self.config.device)
        self.tet_face_tetidx_fx2 = torch.from_numpy(tet_facetet_idx_fx2).long().to(self.config.device)

        # ----------------------------------
        # Initialize neural network
        # ----------------------------------
        # TODO understand all the arguments again
        # Resolved: all of these are model-construction settings coming from the saved config.
        # Even though they look verbose, they should stay because they define the trained architecture.
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

        # ----------------------------------
        # Precompute deformation matrices
        # ----------------------------------
        # Still needed because DefTet uses inverse tetrahedron matrices internally.
        inverse_v = nn.Parameter(self.deftet.tet_inverse_v(
            self.init_tet_pos, self.init_tet_fx4))
        inverse_v.requires_grad = False
        self.deftet.inverse_v = inverse_v.to(self.config.device)

        # ----------------------------------
        # Parallel execution wrapper
        # ----------------------------------
        self.device_count = torch.cuda.device_count()

        # TODO: do I need to keep all these?
        # Resolved: most of these arguments are still required because ParallelWrapper
        # is the entry point that actually performs the forward pipeline.
        self.parallel = SimpleParallelWrapper(
            self.model,
            self.deftet,
            experiment.dir_path('visualization'),
            self.point_adj_sparse,
            self.device_count,
            timing=None,
            use_two_encoder=self.config.use_two_encoder,
            add_input_noise=self.config.add_input_noise,
            n_point=5000 if self.config.res != 100 else 10000,
            use_lap_layer=self.config.use_lap_layer)

        assert self.device_count == 1  # evaluate on one GPU

    def load_pretrain(self, pretrain_path, step=None):
        """
        Load pretrained model weights.
        """

        prefix = 'best_'
        post_fix = ''

        if step is not None:
            post_fix = '_' + str(step)
            prefix = ''

        # Load occupancy decoder
        load_path = os.path.join(pretrain_path, prefix + 'decoder_occ' + post_fix + '.pth')
        load_dict = torch.load(load_path)
        self.model.decoder_occ.load_state_dict(load_dict)

        # Load deformation decoder (if applicable)
        load_path = os.path.join(pretrain_path, prefix + 'decoder_pos' + post_fix + '.pth')
        load_dict = torch.load(load_path)
        if not self.config.baseline:
            self.model.decoder_pos.load_state_dict(load_dict)

        # Load encoder
        load_path = os.path.join(pretrain_path, prefix + 'encoder' + post_fix + '.pth')
        load_dict = torch.load(load_path)
        self.model.encoder.load_state_dict(load_dict)

        # what is this TODO
        # Resolved: this loads the extra Laplacian deformation head, only when that model option was used.
        if self.config.use_lap_layer:
            load_path = os.path.join(pretrain_path, 'lap_decoder_pos.pth')
            load_dict = torch.load(load_path)
            self.model.lap_decoder_pos.load_state_dict(load_dict)

    def inference(self):
        """
        Run inference on a single validation sample.
        Minimal version: no metrics, only forward pass + mesh extraction + optional save.
        """
        self.model.eval()
        with torch.no_grad():
            data = next(iter(self.dataloader_val))

            # ----------------------------------
            # Input data
            # ----------------------------------
            cat = data['synset'][0]

            # Query points are still passed because ParallelWrapper expects them.
            points = data['sdf_point'].float().to(self.config.device)

            # Ground truth surface samples are also still passed because the wrapper expects them.
            surface_point = data['sample_points'].float().to(self.config.device)

            # Ground truth mesh vertices and faces.
            # In the minimal script these are not used for metrics anymore,
            # but ParallelWrapper may still expect them.
            all_verts = [v.to(self.config.device).unsqueeze(0).expand(
                self.device_count, -1, -1) for v in data['verts']]
            all_faces = [v.to(self.config.device).unsqueeze(0).expand(
                self.device_count, -1, -1) for v in data['faces']]

            # ----------------------------------
            # Prepare tetrahedral mesh (batched)
            # ----------------------------------
            # TODO what is meant by batched
            # Resolved: batched means one copy of the initial tet grid per item in the batch.
            # Since batch_size=1 here, this is just adding a batch dimension of size 1.
            #
            # TODO check range and make sure surface points are in the same range where does that happen
            # Resolved: the centering of the tet grid happens here with `-0.5`.
            # The matching data normalization happens upstream in the dataset / preprocessing code,
            # not inside this script.
            init_tet_pos_bxnx3 = self.init_tet_pos.float().unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)

            # TODO make a test file and see why a batch per surface point
            # Resolved: it is not one tet grid per surface point.
            # `surface_point.shape[0]` is batch size, not number of sampled surface points.
            init_tet_bxfx4 = self.init_tet_fx4.unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)

            # Precomputed face-to-tet and face connectivity
            tet_face_tetidx_bxfx2 = self.tet_face_tetidx_fx2.unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)
            init_tet_face_bxfx3 = self.tet_face_fx3.unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)

            # ----------------------------------
            # Forward pass through full pipeline
            # ----------------------------------
            # Outputs include many training/evaluation terms because ParallelWrapper returns them.
            # We keep the call, but ignore the metric-like outputs.
            #
            # TODO can I remove cameras and unnecessary stuff
            # Resolved: yes, because ParallelWrapper was simplified for point-cloud-only inference.
            _, _, _, _, _, _, _, _, _, \
            tet_pos, pred_occ_prob, _, _, pred_surface, _, _ = self.parallel(
                    init_tet_pos_bxnx3=init_tet_pos_bxnx3,
                    init_tet_bxfx4=init_tet_bxfx4,
                    points=points,
                    surface_point=surface_point,
                    tet_face_tetidx_bxfx2=tet_face_tetidx_bxfx2,
                    all_verts=all_verts,
                    all_faces=all_faces,
                    inference=True,
                    return_surf=True,
                    tet_face_bxfx3=init_tet_face_bxfx3,
                    pred_threshold=0.5 if not self.config.use_lap_layer else self.config.lap_threshold,
                    random_seed=0,
            )

            # ----------------------------------
            # Extract predicted surface mesh
            # ----------------------------------
            mesh_v = tet_pos[0, pred_surface[0].reshape(-1)]
            mesh_f = torch.arange(0, mesh_v.shape[0], device=mesh_v.device, dtype=torch.long).reshape(-1, 3)

            # ----------------------------------
            # Logging results
            # ----------------------------------
            print('Category:', cat)
            print('Name:', data['name'][0])
            print('tet vertices:', tet_pos[0].shape)
            print('tets:', self.init_tet_fx4.shape)
            print('surface vertices:', mesh_v.shape)
            print('surface faces:', mesh_f.shape)
            print('tet_occ:', pred_occ_prob[0].shape)

            if mesh_v.shape[0] == 0:
                print('No surface mesh was extracted.')
                return

            if self.save:
                save_name = experiment.dir_path('minimal_inference_outputs')
                save_name = os.path.join(save_name, data['synset'][0])
                if not os.path.exists(save_name):
                    os.makedirs(save_name)

                base_name = data['name'][0].split('/')[-1]
                print('Saving tet mesh')
                print('  tet vertices:', tet_pos[0].shape)
                print('  tets:', self.init_tet_fx4.shape)
                print('  surface faces:', mesh_f.shape)
                print('  tet_occ:', pred_occ_prob[0].shape)
                print('  max tet index:', self.init_tet_fx4.max().item())
                print('  num vertices:', tet_pos[0].shape[0])

                # TODO actually I do not need all this info just keep the stuff without metrics
                # Resolved: only saving the predicted tet grid, connectivity, tet occupancy,
                # extracted surface vertices/faces, and identifiers.
                np.savez_compressed(
                    os.path.join(save_name, base_name + '.npz'),
                    vertices=tet_pos[0].data.cpu().numpy(),
                    tets=self.init_tet_fx4.data.cpu().numpy(),
                    tet_occ=pred_occ_prob[0].data.cpu().numpy(),
                    surf_vertices=mesh_v.data.cpu().numpy(),
                    faces=mesh_f.data.cpu().numpy(),
                    synset=np.array([data['synset'][0]]),
                    name=np.array([data['name'][0]])
                )

                save_mesh(
                    mesh_v.detach().cpu().numpy(),
                    mesh_f.detach().cpu().numpy(),
                    os.path.join(save_name, base_name + '_pred_surface.obj'),
                )

                save_mesh(
                    data['verts'][0].detach().cpu().numpy(),
                    data['faces'][0].detach().cpu().numpy(),
                    os.path.join(save_name, base_name + '_gt_surface.obj'),
                )


def main(experiment, config, model_path, save=False, step=0):

    # This matches the original script.
    config.c_dim = 512

    # We evaluate one by one.
    # TODO: changed dataloader, need to check if it works with the new one
    # Resolved: the script assumes `vox_dataloader.create_dataloader(...)`
    # returns the same dictionary keys used below.
    # If it does not, this is the first place to debug.
    dataloader_val = create_dataloader(batch_size=1, train=False, only_chairs=False)

    print('==> Init Engine')
    trainer = Engine(config=config,
                     dataloader_val=dataloader_val,
                     save=save)

    print('==> Load Pretrain')
    if step == 0:
        step = None
    trainer.load_pretrain(model_path, step=step)

    print('==> Run Inference')
    trainer.inference()


if __name__ == '__main__':
    # TODO what is going on exactly here
    # Resolved:
    # - cudnn.benchmark = True lets cuDNN choose fast kernels for fixed input sizes
    # - get_parser() reads command line arguments
    # - Experiment.load(...) restores the saved experiment config/state
    # - experiment_id/root_path rebuild output directory structure
    # - dataset_dir manually points the config to the dataset location
    # - main(...) runs model loading + one-sample inference
    torch.backends.cudnn.benchmark = True
    args = get_parser()
    experiment = Experiment.load(args.experiment_path, options=OPTIONS)
    experiment.experiment_id = args.experiment_path.split('/')[-1]
    experiment.config.dataset_dir = '/work3/s233736/datasets/mesh_surfaces' #'/work3/s233736/datasets/MRI'  # TODO check if this is needed, should be in the saved config already
    experiment.root_path = os.path.join(DEFAULT_FOLDER_PATH, experiment.experiment_id)
    config = experiment.config
    main(experiment, config, args.experiment_path,
         save=args.save, step=args.step)
