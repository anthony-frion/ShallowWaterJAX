import hashlib
import logging
import pickle
import time
import os
from typing import Union
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils.utilities as ut
from torch.optim.lr_scheduler import StepLR
logger = logging.getLogger(__name__)
import json
import socket
import datetime
import os
import re


class Assimilation(torch.nn.Module):
    def __init__(
        self,
        emulator: Union[torch.nn.Module, pl.LightningModule],
        observations,
        observations_operator,
        initial_state,
        dt_list,
        physical_params = None,
        other_params = None,
        device: str = "cpu",
        optimizers: Union[dict, list]=None,
        loss_function=None,
        max_steps: int =1000,
        switch_to_lbfgs_after: int = 1000,
        ckp_path: str = None,
    ):
        super(Assimilation, self).__init__()

        self.emulator = emulator
        self.observations = observations
        self.observations_operator = observations_operator
        self.initial_state = initial_state
        self.physical_params = physical_params
        self.other_params = other_params
        self.dt_list = dt_list
        self.device = device
        self.max_steps = int(max_steps)
        self.lbfgs_steps = int(max_steps - switch_to_lbfgs_after)

        if optimizers is not None:
            isinstance(optimizers, (dict, list)), "optimizers should be a dictionary or a list"
            if isinstance(optimizers, dict):
                for key in optimizers.keys():
                    if key != "adam" and key != "lbfgs":
                        raise ValueError(f"{key} optimizer should be adam or lbfgs")
                self.optimizers = optimizers
        else:
            self.optimizers = {"adam": None, "lbfgs": None}

        if loss_function is not None:
            assert callable(loss_function), "loss_function should be a function"
            self.loss_function = loss_function
        else:
            self.loss_function = torch.nn.MSELoss()

        self.loss = []
        self.switch_to_lbfgs_after = int(switch_to_lbfgs_after)
        
        self.scheduler_adam = None
        self.ckp_path = ckp_path
        self.results = {'x0': [], 'params': [], 'loss': []}
        self.writer = None
    @staticmethod
    def freeze_model(emulator):
        emulator.model.eval()  # Should we use set it to eval()?
        for param in emulator.model.parameters():
            param.requires_grad = False  # Switch off the gradient computation
        return emulator


    def generate_pseduo_obseravation_data(self):
        ...
        self.observations = ...
        return self.observations
    
    # def set_optimizers(self, params):
    #     if "adam" in self.optimizers.keys():
    #         self.optimizers["adam"] = torch.optim.Adam(params=params, lr=1e-2)
    #         self.scheduler_adam = StepLR(self.optimizers["adam"], step_size=self.max_steps/4, gamma=0.5)
    #     if "lbfgs" in self.optimizers.keys():
    #         self.optimizers["lbfgs"] = torch.optim.LBFGS(params=params, lr=0.5, max_iter=30, line_search_fn='strong_wolfe')

        # optimizer_adam = torch.optim.Adam([self.initial_state], lr=1e-2)
        # optimizer_lbfgs = torch.optim.LBFGS(params=[self.initial_state], lr=0.005, max_iter=50, line_search_fn='strong_wolfe')
        # return {"adam": optimizer_adam, "lbfgs": optimizer_lbfgs}
        # return torch.optim.LBFGS(params=[self.initial_state], lr=0.1, max_iter=20, line_search_fn='strong_wolfe')
        
    def initialize_optimizer(self):
        if self.initial_state is None:
            raise ValueError("initial_state must be initialized before calling initialize_optimizer.")
              
        params = [self.initial_state]
        params.append(self.physical_params) if self.physical_params is not None else None
        
        if "adam" in self.optimizers.keys():
            self.optimizers["adam"] = torch.optim.Adam(params=params, lr=1e-2)
            self.scheduler_adam = StepLR(self.optimizers["adam"], step_size=self.max_steps/4, gamma=0.5)
        if "lbfgs" in self.optimizers.keys():
            self.optimizers["lbfgs"] = torch.optim.LBFGS(params=params, lr=0.5, max_iter=self.lbfgs_steps, line_search_fn='strong_wolfe',
                                                         tolerance_grad=1e-10, tolerance_change=1e-12)
    
  
    def test_emulator(self, dataloader_val):
        self.emulator.eval()
        for i, data in enumerate(dataloader_val):
            if self.physical_params is not None:
                x, y, params = data
                params = params.to(self.device)
            else:
                x, y = data
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                if self.physical_params is not None:
                    y_hat = self.emulator(x, params)
                else:
                    y_hat = self.emulator(x)
                ut.plot_x_2d(y_hat[0,0,...], 'emulated')
                ut.plot_x_2d(y[0,0,...], 'observed')
                ut.plot_x_2d(y_hat[0,0,...] - y[0,0,...], 'difference')
        return None
    
    def _convert_and_append(self, param):
        if isinstance(param, np.ndarray):
            param = torch.from_numpy(param).to(self.device)
            param.requires_grad = True
            return param
        return None

    def set_initState_and_physParams(self):
        # converted_params = []
        for i, param in enumerate([self.initial_state, self.physical_params]):
            converted = self._convert_and_append(param)
            # converted_params.append(converted)
            if i == 0:  # initial_state
                self.initial_state = converted
            elif i == 1:  # physical_params
                self.physical_params = converted
        # return converted_params
    
    def set_summary_writer(self):
        # name of saving model would be here
        machine_name = socket.gethostname()
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # Using tensorboard
        EXP_NAME = os.path.basename(self.ckp_path)
        writer = SummaryWriter(log_dir=f"runs/inv_{current_datetime}_{machine_name}_{EXP_NAME}/")
        return writer

    def run_data_assimilation(self, datatest_loader):
        # Test the emulator
        # self.test_emulator(datatest_loader)
        # return 1, 2
        # Using tensorboard
        self.writer = self.set_summary_writer()
        # freeze the model
        self.freeze_model(self.emulator)
        # set the initial state
        # self.initial_state = torch.from_numpy(self.initial_state).to(self.device)
        # self.initial_state.requires_grad = True
        self.set_initState_and_physParams()
        # set the optimizer
        self.initialize_optimizer()
        # self.set_optimizers(x0_and_params)
        # Define the switching criterion (e.g., after 100 steps)
        # self.switch_to_lbfgs_after = 400
        # for step in tqdm(range(self.max_steps)):
        optimizer = self.optimizers["adam"]
        for iter in tqdm(range(self.switch_to_lbfgs_after)):
            # Use Adam for the initial phase
            # optimizer = self.optimizers["adam"]
            optimizer.zero_grad()
            simulations = self.rollout(self.emulator, self.initial_state, self.physical_params, self.other_params, further_step=self.dt_list)
            loss = self.observations_operator.mseloss(simulations, self.observations, self.dt_list)
            loss.backward()
            optimizer.step()
            self.scheduler_adam.step() # Update the scheduler
            self._update_writer(loss, iter, opt='adam')
        # Switch to LBFGS for fine-tuning
        optimizer = self.optimizers["lbfgs"]
        # loss = 0
        iter = [self.switch_to_lbfgs_after]

        class OptimizationStop(Exception):
            pass

        def closure():
            optimizer.zero_grad()
            simulations = self.rollout(self.emulator, self.initial_state, self.physical_params, self.other_params, further_step=self.dt_list)
            # ut.plot_x_2d(self.observations[0,0,...], 'observed')
            loss = self.observations_operator.mseloss(simulations, self.observations, self.dt_list)            
            # print(f"Loss at LBFGS sub-step {step}: {loss.item()}")
            loss.backward()
            self._update_writer(loss, iter[0], opt='lbfgs')
            iter[0] += 1
            if self._check_criteria(loss):
                # print("Loss is less than 1e-10, stopping early.")
                # GO OUT OF THE CLOSURE
                raise OptimizationStop("Loss criterion reached. Stopping LBFGS optimization.")
            return loss
        try:
            optimizer.step(closure=closure)
        except OptimizationStop as e:
            print(e)
            #     ut.save_data(f"{self.ckp_path}/x0_tuning_{step}.pkl", self.initial_state.detach().cpu().numpy())
            #     ## remove the old symlink
            #     if os.path.exists(f"{self.ckp_path}/x0.pkl"):
            #         os.remove(f"{self.ckp_path}/x0.pkl")
            #     os.symlink(f"x0_tuning_{step}.pkl", f"{self.ckp_path}/x0.pkl", target_is_directory=False)
            #     if self.physical_params is not None:
            #         ut.save_data(f"{self.ckp_path}/params_tuning_{step}.pkl", self.physical_params.detach().cpu().numpy())
            #         ## remove the old symlink
            #         if os.path.exists(f"{self.ckp_path}/params.pkl"):
            #             os.remove(f"{self.ckp_path}/params.pkl")
            #         os.symlink(f"params_tuning_{step}.pkl", f"{self.ckp_path}/params.pkl", target_is_directory=False)
        self.writer.flush()
        self.writer.close()
        # save the loss history with json
        # with open(f"{self.ckp_path}/loss_history.json", "w") as f:
        #     json.dump(self.loss, f)
        ut.save_values_with_arrays(f"{self.ckp_path}/x0_params.pkl", self.results)
        return self.initial_state, self.physical_params, self.loss

    def _check_criteria(self, loss):
        eps = loss.detach().cpu().item()
        if eps < 1e-10:
            print("Loss is less than 1e-10, stopping early.")
            return True
        return False
    
    def _update_writer(self, loss, iter, opt):
        # Common steps for both optimizers
        current_loss = loss.detach().cpu().item()
        self.loss.append(current_loss)
        # print the loss
        print(f"Loss at step {iter} using {opt}: {current_loss}")
        # write the loss to tunning log in tensorboard
        self.writer.add_scalar("Loss/tuning", current_loss, iter)
        self.writer.add_scalar("scheduler LR", self.scheduler_adam.get_last_lr()[0], iter)

        # check if the loss is less than 1e-8
        # if current_loss < 1e-10:
        #     print("Loss is less than 1e-10, stopping early.")
        #     break
        self.results['loss'].append(current_loss)
        if iter % 10 == 0:
            # Save the model
            self.results['x0'].append(self.initial_state.detach().cpu().numpy())
            if self.physical_params is not None:
                self.results['params'].append(self.physical_params.detach().cpu().numpy())

    def rollout(self, emulator, initial_state, params, others, further_step: Union[int, list] = 1):
        assert isinstance(further_step, (int, list)), "further_step must be either an integer or a list"
        further_step = [further_step] if isinstance(further_step, int) else further_step
        emulator.model.eval()
        states = []
        state = initial_state.clone()
        if re.search(r'swe', emulator.problem_name, re.IGNORECASE):
            H = others['H'].to(self.device)
            mask = others['mask'].to(self.device)
            
        for step in further_step:
            # print(f"Rollout step: {step}")
            if step == 0:
                states.append(state)
                continue
            # Simplify the loop by directly using the last state if available
            for _ in range(step if not states else 1):
                if re.search(r'swe', emulator.problem_name, re.IGNORECASE):
                    # X, X_next_target, params, H, mask = batch
                    state = emulator.forward(state, params=params, H=H, mask=mask)
                # elif re.search(r'tsunami', self.problem_name, re.IGNORECASE):
                else:
                    if self.emulator.model.condi_net:
                        state = emulator.forward(state, params)
                    else:
                        state = emulator.forward(state)
            states.append(state)

        # print("len(states): ", len(states))
        return ut.list2tsr(states, dim=0)
