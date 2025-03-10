import numpy as np
import torch
import torch.optim as optim
# import pytorch_lightning as pl


import utils.utilities as ut
import models.models as m
import models.uncondi_models as um

import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from dataset.dataset_nemo import MyDatasetNEMO2D, preprocess_data

import models.lightning_models as lm
import xarray as xr
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import matplotlib.pyplot as plt
from src.nnUtils import autoregressive, rollout
import src.cal_inverse as inverse
from models import networks as net
from models.emulator_module import EmulatorModule
# from pytorch_lightning.loggers import TensorBoardLogger
from lightning import seed_everything
from lightning.pytorch import loggers as pl_loggers
import lightning.pytorch as pl
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import LearningRateFinder, ModelCheckpoint
from torch.profiler import ProfilerActivity
from src.assimilation import Assimilation
import src.observation as obs
from models import time_integration

import datetime
import socket
import os
import glob
import shutil

# A logger for this file
log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")


torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

# Adding hydra configuration herer
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def run_data_assimilation(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(f"============== TRAINING MODEL for TSUNAMI TESTCASE ==============")
    DATA_DIR            = cfg.paths.data_dir
    TESTCASE            = cfg.experiment.inverse.data.testcase
    FILENAME            = cfg.experiment.inverse.data.file_name
    SAVE_INV_DIR        = cfg.paths.save_inv_dir
    EXP_NAME            = cfg.experiment.inverse.name
    WK_DIR              = cfg.paths.wk_dir
    NETWORK             = cfg.experiment.inverse.network.model
    CONDI_NET           = cfg.experiment.inverse.network.condi_net
    FORWARD_MODEL       = cfg.experiment.inverse.forward_model.name
    DT                  = cfg.experiment.inverse.forward_model.dt
    PATH_TO_FILE        = f"{DATA_DIR}/{TESTCASE}/{FILENAME}"
    INV_EXP_DIR         = f"{SAVE_INV_DIR}/{EXP_NAME}"
    # case                = cfg.experiment.inverse.data.case
    spin_up_time        = cfg.experiment.inverse.args.spin_up_time
    num_obs             = cfg.experiment.inverse.args.num_obs
    pathfile            = f"{WK_DIR}/{cfg.experiment.inverse.load_model.path}/{cfg.experiment.inverse.load_model.name}"
    machine_name        = socket.gethostname()
    current_datetime    = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tsb_path            = f"./runs/inverse_{current_datetime}_{machine_name}_{EXP_NAME}"
    NORMALIZE           = cfg.experiment.inverse.data.normalize
    NORMALIZE_TYPE      = cfg.experiment.inverse.data.normalize_type
    CONSTRAINT_BC       = cfg.experiment.inverse.network.constraint_bc
    print("Path to file:", PATH_TO_FILE)
    print("Running on:", cfg.resource.partition)
    print("+++++++++++++++ Running assimilation +++++++++++++++")
    # Create inverse directory if it doesn't exist
    
    os.makedirs(INV_EXP_DIR, exist_ok=True)
    print(f"Inverse directory created: {INV_EXP_DIR}")
    yaml_file_path = os.path.join(WK_DIR, "conf/experiment/inverse", f"{EXP_NAME}.yaml")
    # Create the YAML file to the inverse directory
    shutil.copy(yaml_file_path, INV_EXP_DIR)
    print(f"YAML file copied to {INV_EXP_DIR}")

    model_params = {
        'name': cfg.experiment.inverse.network.model,
        'condi_net': cfg.experiment.inverse.network.condi_net,
        'in_channels': cfg.experiment.inverse.network.in_channels,
        'out_channels': cfg.experiment.inverse.network.out_channels,
        'features': cfg.experiment.inverse.network.features,
        'kernel_size': cfg.experiment.inverse.network.kernel_size,
        'stride': cfg.experiment.inverse.network.stride,
        'padding': cfg.experiment.inverse.network.padding,
        'norm_type': cfg.experiment.inverse.network.norm_type,
        'activation': cfg.experiment.inverse.network.activation,
        'num_layers': cfg.experiment.inverse.network.num_layers,
    }

    # for i in case:
    #     print(f"{DATA_DIR}/{TESTCASE}/{FILENAME}-{i:003}.nc")
    # ds = xr.open_mfdataset([f"{DATA_DIR}/{TESTCASE}/{FILENAME}-{i:003}.nc" for i in case],
    #                     concat_dim='combined_index', combine='nested')
    # # ds = xr.open_mfdataset([f"{cfg.paths.data_dir}/{cfg.params.testcase}/{cfg.files.data_name}-{i:003}.nc" for i in range(1, cfg.test.datafile_size + 1)],
    # #                        concat_dim='combined_index', combine='nested')
    # ssh = ds['ssh'].values  # sea-surface height - (time_counter, y_grid_T, x_grid_T)
    # uos = ds['baro_u'].values  # zonal velocity - (time_counter, y_grid_U, x_grid_U)
    # vos = ds['baro_v'].values  # meridional velocity - (time_counter, y_grid_V, x_grid_V)

    # print("uos.shape = ", uos.shape)
    # print("vos.shape = ", vos.shape)
    # print("ssh.shape = ", ssh.shape)

    # if CONDI_NET:
    #     # conditional learning
    #     # --------------- This is only for conditional network ----------------
    #     zdom, cD  = ut.get_depth_and_cD_TSUNAMI(data_log=f"./data/{TESTCASE}/data_generating.log",
    #                                     case=case,
    #                                     data_size=ssh.shape[0],
    #                                     nt=ssh.shape[1],
    #                                     ny=ssh.shape[2],
    #                                     nx=ssh.shape[3])
    #     full_dataset = MyDatasetNEMO2D(ssh, uos, vos, params=dict(zdom=zdom, cD=cD), normalize=True)
    #     # print zdom of 1st testcase
    #     print("zdom of 1st testcase", zdom)
    #     # print cD of 1st testcase
    #     print("cD of 1st testcase", cD)
    # else:
    #     # Unconditional learning
    #     full_dataset = MyDatasetNEMO2D(ssh, uos, vos, params=None, normalize=True)

    ocean_ds, ssh, params = preprocess_data(np.load(PATH_TO_FILE), TESTCASE, CONDI_NET, NORMALIZE, NORMALIZE_TYPE)

    obs_idx_1, obs_idx_2 = ut.generate_indices(len(ocean_ds), train_frac=1.0)
    obs_sampler_1, obs_sampler_2 = ut.generate_samplers(obs_idx_1, obs_idx_2)
    # obs_set_1 = full_dataset[obs_idx_1]
    # obs_set_2 = full_dataset[obs_idx_2]
    dataloader_obs_1 = DataLoader(dataset=ocean_ds, shuffle=False, batch_size=len(obs_idx_1), sampler=obs_sampler_1)
    # dataloader_obs_2 = DataLoader(dataset=full_dataset, shuffle=False, batch_size=len(obs_idx_2), sampler=obs_sampler_2)
    # Unpack the values with a placeholder for params which might not always be used
    obs_input, obs_target, *optional_params = next(iter(dataloader_obs_1))
    physical_params = optional_params[0] if CONDI_NET and optional_params else None

    print("obs_input = ", obs_input.shape)
    obs_input = obs_input[spin_up_time:spin_up_time+num_obs, ...]
    obs_target = obs_target[spin_up_time:spin_up_time+num_obs, ...]
    physical_params = physical_params[spin_up_time:spin_up_time+num_obs, ...] if physical_params is not None else None
    
    print("obs_input = ", obs_input.shape)
    # return
    DEVICE = "cuda" if cfg.resource.partition == "gpu" else "cpu"
    model = getattr(net, NETWORK)(**model_params)
    # model = getattr(net, NETWORK)(condi_net=CONDI_NET)
    model = ut.load_model(model, pathfile, device=DEVICE)
    model = model.to(DEVICE)

    emulator = EmulatorModule(
            problem_name=TESTCASE,
            model=model,
            forward_model_name=FORWARD_MODEL,
            dt=DT,
            optimizer=None,
            scheduler=None,
            data_module=ocean_ds,
            constraintBC=CONSTRAINT_BC
    )

    time_obs = list(range(spin_up_time, spin_up_time+num_obs, 1))
    nt = len(time_obs)
    nb = 1
    ny = 100
    nx = 100
    print("time_obs = ", time_obs)
    # return
    obs_op = obs.MaskedGaussianNoise(  # we never observe u,v. we observe exactly 25% of locations at each time step for h values with noise s.d. of 1 meter 
        p_obs=dict(
            h=cfg.experiment.inverse.obs.h,
            u=cfg.experiment.inverse.obs.u,
            v=cfg.experiment.inverse.obs.v,
        ),
        noise_sd=cfg.experiment.inverse.obs.noise_sd,
        gridsize=dict(h=[nt, nb, ny, nx], u=[nt, nb, ny, nx], v=[nt, nb, ny, nx]),
        t_axis=time_obs,
        FixedObservationCount=True,
        device=DEVICE)
    
    pseudoobs = obs_op.observe(obs_input)
    PLOT = False
    if PLOT == True:
        for i in range(num_obs):
            ut.plot_x_2d(pseudoobs[i, 0, ...], f"pseudoobs at time {time_obs[i]}")
            ut.plot_x_2d(obs_input[i, 0, ...], f"obs_input at time {time_obs[i]}")
    # return

    # initial guess is generated randomly
    np.random.seed(128)
    initial_guess = np.random.rand(1, 3, ny, nx).astype(np.float32)
    if CONDI_NET:
        params_guess = np.random.rand(1, 2).astype(np.float32)
    else:
        params_guess = None
    if ocean_ds.problem == 'swe':
        other_params = dict(H=optional_params[1][0:1, ...],
                            mask=optional_params[2][0:1, ...]
                            )
    # ut.plot_x_2d(initial_guess[0, 0], "input initial state")
    # initial_guess = torch.from_numpy(initial_guess).to(device=DEVICE)

    tuning_obj = Assimilation(emulator=emulator,
                    observations=pseudoobs,
                    observations_operator=obs_op,
                    initial_state=initial_guess,
                    dt_list=obs_op.t_axis,
                    physical_params=params_guess,
                    other_params=other_params,
                    device=DEVICE,
                    optimizers={"adam": None, "lbfgs": None},
                    loss_function=None,
                    max_steps=cfg.experiment.inverse.args.max_steps,
                    switch_to_lbfgs_after=cfg.experiment.inverse.args.switch_to_lbfgs_after,
                    ckp_path=INV_EXP_DIR
                    )
    
    x0, params, loss = tuning_obj.run_data_assimilation(dataloader_obs_1)

if __name__ == '__main__':
    run_data_assimilation()