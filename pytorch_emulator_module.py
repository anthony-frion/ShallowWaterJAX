from typing import Any
# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
from models import time_integration
import utils.utilities as ut
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re

class EmulatorModule(pl.LightningModule):
    def __init__(self,
                 problem_name: str, 
                 model, 
                 forward_model_name: str, 
                 dt:float, 
                 loss_function=None, 
                 optimizer=None, 
                 scheduler=None, 
                 save_ckp=None, 
                 data_module=None,
                 constraintBC=False, 
                 data_type=torch.float32,
                 callbacks=None):
        super(EmulatorModule, self).__init__()
        
        # Check model is a network, otherwise raise an error
        assert isinstance(model, torch.nn.Module), "model should be a network"

        self.problem_name = problem_name
        self.model = model

        self.data_module = data_module

        self.constraintBC = constraintBC
        self.data_type = data_type

        self.callbacks = callbacks

        if type(forward_model_name) is str:
            # Get the class from the forward_model_name string
            # Check the forward_model_name is a valid class, with valid string name like "ForwardEuler", "RK2,"RK4", otherwise raise an error
            assert forward_model_name in ["ForwardEuler", "RK2", "RK4", "ForwardEulerConstraintBC"], "forward_model_name should be one of 'ForwardEuler', 'RK2', 'RK4'"

            # Create an instance of the class
            self.forward_model = getattr(time_integration, forward_model_name)(model, dt)
        else:
            self.forward_model = forward_model_name
       
        # Check the loss function is a function, otherwise raise an error
        if loss_function is not None:
            assert callable(loss_function), "loss_function should be a function"
            self.loss_function = loss_function
        else:
            self.loss_function = torch.nn.MSELoss()

        if optimizer is not None:
            assert isinstance(optimizer, torch.optim.Optimizer), "optimizer should be an instance of torch.optim.Optimizer"
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters())
        
        
        self.scheduler = scheduler
        if save_ckp is not None:
            self.save_ckp = save_ckp
        else:
            self.save_ckp = "model.pth.tar"

        self.train_outputs = {}
        self.val_outputs = {}

    # def forward(self, X, params=None, H=None, maskBC=None):
    #     X_next = self.forward_model.step(X, params=params, H=H, maskBC=maskBC)
    #     # if re.search(r'swe', self.problem_name, re.IGNORECASE):
    #     #     X_next = X_next * maskBC
    #     return X_next
    
    def forward(self, X, params=None, *args, **kwargs):
        X_next = self.forward_model.step(X, params, *args, **kwargs)
        if self.constraintBC:
            if re.search(r'swe', self.problem_name, re.IGNORECASE):
                X_next[:, 0, 0, :] = X_next[:, 0, -1, :] = \
                X_next[:, 0, :, 0] = X_next[:, 0, :, -1] = - torch.Tensor([self.data_module.normalize['ssh']['mean'] / self.data_module.normalize['ssh']['std']])
                X_next[:, 1, 0, :] = X_next[:, 1, -1, :] = \
                X_next[:, 1, :, 0] = X_next[:, 1, :, -1] = - torch.Tensor([self.data_module.normalize['u']['mean'] / self.data_module.normalize['u']['std']])
                X_next[:, 2, 0, :] = X_next[:, 2, -1, :] = \
                X_next[:, 2, :, 0] = X_next[:, 2, :, -1] = - torch.Tensor([self.data_module.normalize['v']['mean'] / self.data_module.normalize['v']['std']])
        return X_next
    # def forward(self, X, params=None):
    #     return self.forward_model.forward(x)

    def training_step(self, batch, batch_idx):
        # X, X_next_target, params, H, maskBC = batch
        if re.search(r'swe', self.problem_name, re.IGNORECASE):
            X, X_next_target, params, H, mask = batch
            X_next_predict = self.forward(X, params=params, H=H, mask=mask)
        # elif re.search(r'tsunami', self.problem_name, re.IGNORECASE):
        else:
            if self.model.condi_net:
                X, X_next_target, params = batch
            else:
                X, X_next_target = batch
                params = None
            X_next_predict = self.forward(X, params=params)
        # if self.model.condi_net:
        #     X, X_next_target, params = batch
        # else:
        #     X, X_next_target = batch
        #     params = None
        # X_next_predict = self.forward(X, params, H, maskBC)
        loss = self.loss_function(X_next_target, X_next_predict)
        
        # --------------------- Logging -------------------------------
        # Log the loss value in scientific notation
        # self.log('Loss/Train', loss, on_step=True, prog_bar=True, logger=True, 
            #  on_epoch=True, sync_dist=True)
        
        #  Log the current learning rate
        # lr = self.optimizer.param_groups[0]['lr']
        # self.log("lr", lr, on_step=True, prog_bar=True, logger=True, 
        #      on_epoch=True) # , sync_dist=True)

        # Save the loss as an instance attribute
        if self.current_epoch not in self.train_outputs:
            self.train_outputs[self.current_epoch] = []
        self.train_outputs[self.current_epoch].append(loss.detach().cpu().item())
        # --------------------- END Logging -------------------------------

        return loss
    
    def on_train_epoch_end(self) -> None:
        # Access the outputs from train outputs
        outputs = self.train_outputs[self.current_epoch]
        avg_loss = torch.tensor(outputs).mean()
        self.log("avg_training_loss", avg_loss, prog_bar=True)
        #  Log the current learning rate
        lr = self.optimizer.param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True)
        
        with open("train_losses.out", "a") as f:
            f.write(f"Epoch {self.current_epoch}: {avg_loss} - lr: {lr}\n")
            # f.write(f"{self.current_epoch}: {avg_loss}\n")


    def validation_step(self, batch, batch_idx):
        # X, X_next_target, params, H, maskBC = batch
        if re.search(r'swe', self.problem_name, re.IGNORECASE):
            X, X_next_target, params, H, mask = batch
            X_next_predict = self.forward(X, params=params, H=H, mask=mask)
        # if re.search(r'tsunami', self.problem_name, re.IGNORECASE):
        else:
            if self.model.condi_net:
                X, X_next_target, params = batch
            else:
                X, X_next_target = batch
                params = None
            # if params is not None:  
            #     X, X_next_target, params = batch
            # else:
            #     X, X_next_target = batch
            #     params = None
            X_next_predict = self.forward(X, params=params)
        # if self.model.condi_net:
        #     X, X_next_target, params = batch
        # else:
        #     X, X_next_target = batch
        #     params = None
        # if self.problem_name == "swe":
        #     X_next_predict = self.forward(X, params=params, H=others[0], mask=others[1])
        # if self.problem_name == "tsunami":
        #     X_next_predict = self.forward(X, params=params)
        loss = self.loss_function(X_next_target, X_next_predict)

        # --------------------- Logging -------------------------------
        # Log the loss value in scientific notation
        # self.log('Loss/Val', loss, on_step=True, prog_bar=True, logger=True, 
            #  on_epoch=True, sync_dist=True)
        
        # Save the loss as an instance attribute
        if self.current_epoch not in self.val_outputs:
            self.val_outputs[self.current_epoch] = []
        self.val_outputs[self.current_epoch].append(loss.detach().cpu().item())
        # --------------------- END Logging -------------------------------
        
        # return loss

    def on_validation_epoch_end(self):
        # Access the outputs from validation_step
        outputs = self.val_outputs[self.current_epoch]
        avg_loss = torch.tensor(outputs).mean()
        self.log("avg_val_loss", avg_loss, prog_bar=True)
        with open("val_losses.out", "a") as f:
            f.write(f"Epoch {self.current_epoch}: {avg_loss}\n")
        # # This may cause error or running slow or not working
        # dir_name = os.path.dirname(self.save_ckp)
        # file_name = os.path.basename(self.save_ckp)
        # symlink_path = f"{dir_name}/model.pth.tar"
        # print("self.save_ckp = ", self.save_ckp)
        # # Create a new symbolic link
        # try:
        #     os.symlink(file_name, symlink_path, target_is_directory=False)
        # except FileExistsError:
        #     os.remove(symlink_path)
        #     os.symlink(file_name, symlink_path, target_is_directory=False)  
        #  # # This may cause error or running slow or not working
        dir_name = os.path.dirname(self.save_ckp)
        file_name = os.path.basename(self.save_ckp)
        symlink_path = f"{dir_name}/model.pth.tar"
        print("--- Creating symbolic link ---")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # Remove existing symlink if it exists, then create a new one
        create_symlink(file_name, symlink_path)
        # # Create a new symbolic link
        # try:
        # #     os.symlink(file_name, symlink_path, target_is_directory=False)
        #     subprocess.run(['ln', '-s', file_name, symlink_path], check=True, cwd=dir_name)
        #     print("--- Symbolic link created successfully ---")
        # except FileExistsError:
        #     subprocess.run(['rm', symlink_path], check=True, cwd=dir_name)
        #     subprocess.run(['ln', '-s', file_name, symlink_path], check=True, cwd=dir_name)
        #     print("--- Symbolic link created successfully ---")
        

    # def on_validation_epoch_end(self) -> None:
    #     outputs = self.trainer.callback_metrics
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     with open("val_losses.txt", "a") as f:
    #         f.write(f"{self.current_epoch}: {avg_loss}\n")
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # X, X_next_target, params, H, maskBC = batch
        if re.search(r'swe', self.problem_name, re.IGNORECASE):
            X, X_next_target, params, H, mask = batch
            X_next_predict = self.forward(X, params=params, H=H, mask=mask)
        # if re.search(r'tsunami', self.problem_name, re.IGNORECASE):
        else:
            if self.model.condi_net: 
                X, X_next_target, params = batch
            else:
                X, X_next_target = batch
                params = None
            print("X.shape = ", X.shape)
            print("X_next_target.shape = ", X_next_target.shape)
            print("params = ", params)
            X_next_predict = self.forward(X, params=params)
        # if self.model.condi_net:
        #     X, X_next_target, params = batch
        # else:
        #     X, X_next_target = batch
        #     params = None
        # if self.problem_name == "swe":
        #     X_next_predict = self.forward(X, params=params, H=others[0], mask=others[1])
        # if self.problem_name == "tsunami":
        #     X_next_predict = self.forward(X, params=params)
        # X_next_predict = self.forward(X, params, H, maskBC)

        X_next_predict_np = X_next_predict.detach().cpu().numpy()
        X_next_target_np = X_next_target.detach().cpu().numpy()

        # Plot the X_next_predict_np and X_next_target_np
        if self.problem_name == "lorenz96":
            # ploting nicely the first data of x
            plt.figure(figsize=(8, 2))
            plt.imshow(X_next_target_np[:, 0, :].T, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.xlabel('Time step')
            plt.ylabel('State variable')
            plt.title("Simulator")

            plt.figure(figsize=(8, 2))
            plt.imshow(X_next_predict_np[:, 0, :].T, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.xlabel('Time step')
            plt.ylabel('State variable')
            plt.title("Emulator Predicted")

            # error
            plt.figure(figsize=(8, 2))
            plt.imshow(np.abs(X_next_predict_np[:, 0, :].T - X_next_target_np[:, 0, :].T), aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.xlabel('Time step')
            plt.ylabel('State variable')
            plt.title("Error")
            plt.show()
        else:
            plt.figure()
            plt.imshow(X_next_predict_np[0, 0, ...])
            plt.colorbar()
            plt.title("Predicted SSH")
            # plt.savefig("predicted_ssh.png")
            plt.figure()
            plt.imshow(X_next_target_np[0, 0, ...])
            plt.colorbar()
            plt.title("Target SSH")
            plt.show()

            plt.figure()
            plt.imshow(np.abs(X_next_predict_np[0, 0, ...] - X_next_target_np[0, 0, ...]))
            plt.colorbar()
            plt.title("Error of SSH")
            plt.show()
            # ut.plot_result(X_next_predict_np, X_next_target_np, self.test_data_idx, further_step=1, plot='ssh')


    def on_save_checkpoint(self, *args, **kwargs):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # Save the model
        ut.save_model(self.model, filename=self.save_ckp, train_loss=self.train_outputs, val_loss=self.val_outputs)
        # except FileExistsError:
        #     os.remove(symlink_path)
        #     os.symlink(file_name, symlink_path, target_is_directory=False)

        # ut.save_model(self.model, filename=f"{checkpoint_dir}/model.pth.tar", train_loss=self.train_outputs, val_loss=self.val_outputs)
        # symlink_path = f"{checkpoint_dir}/model.pth.tar"
        # Check if the symbolic link already exists
        # if os.path.exists(symlink_path) and os.path.islink(symlink_path):
            # os.remove(symlink_path)  # Remove it if it exists
        # Create a new symbolic link
        # os.symlink(os.path.basename(self.save_ckp), symlink_path, target_is_directory=False)

       

    # def on_save_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
    #     return super().on_save_checkpoint(checkpoint)
    
    def configure_optimizers(self, *args, **kwargs):
        optimizer = self.optimizer
        return [optimizer],  [{"scheduler": self.scheduler, "interval": "epoch"}]


    


    # def setup(self, stage):
    #     if self.model.load_ae_weights:
    #     checkpoint = torch.load(
    #         self.pretrained_checkpoint_path, map_location=self.device
    #     )
    #     # Prepare and load encoder weights
    #     encoder_weights = {
    #         key.replace("encoder.", ""): value  # Adjust key names for encoder
    #         for key, value in checkpoint["state_dict"].items()
    #         if key.startswith("encoder.")
    #     }
    #     self.encoder.load_state_dict(encoder_weights, strict=True)

    #     # Prepare and load decoder weights
    #     decoder_weights = {
    #         key.replace("decoder.", ""): value  # Adjust key names for decoder
    #         for key, value in checkpoint["state_dict"].items()
    #         if key.startswith("decoder.")
    #     }
    #     self.decoder.load_state_dict(decoder_weights, strict=True)
    #     if (
    #     self.training_mode in ("roll_out", "roll_with_grads")
    #     and self.ldae_retrain_config.load_resnet_weights
    #     ):
    #     checkpoint = torch.load(
    #         self.pretrained_checkpoint_path, map_location=self.device
    #     )
    #     # Prepare and load resnet weights
    #     resnet_weights = {
    #         key.replace("resnet.", ""): value  # Adjust key names for resnet
    #         for key, value in checkpoint["state_dict"].items()
    #         if key.startswith("resnet.")
    #     }
    #     self.resnet.load_state_dict(resnet_weights, strict=True)

    #     # Prepare and load flux_layer_x weights
    #     flux_layer_x_weights = {
    #         key.replace(
    #             "flux_layer_x.", ""
    #         ): value  # Adjust key names for flux_layer_x
    #         for key, value in checkpoint["state_dict"].items()
    #         if key.startswith("flux_layer_x.")
    #     }
    #     self.flux_layer_x.load_state_dict(flux_layer_x_weights, strict=True)

    # def configure_optimizers(self):
    #     LEARNING_RATE = 0.001
    #     NUM_EPOCHS = 20
    #     optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    #     # Define learning rate scheduler
    #     scheduler = lr_scheduler.MultiStepLR(
    #         optimizer,
    #         milestones=[int(NUM_EPOCHS - NUM_EPOCHS / i) for i in range(2, 10, 2)],
    #         gamma=0.5,
    #     )
    #     return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


# from pytorch_lightning.callbacks import ProgressBar
# from pytorch_lightning.callbacks import Callback

# class CustomProgressBar(ProgressBar):
#     def init_sanity_tqdm(self):
#         bar = super().init_sanity_tqdm()
#         bar.set_description('Running Sanity Check')
#         bar.set_postfix(refresh=False)
#         return bar

#     def init_train_tqdm(self):
#         bar = super().init_train_tqdm()
#         bar.set_description('Training')
#         bar.set_postfix(refresh=False)
#         return bar

# class CustomFormatter(Callback):
#     def on_validation_end(self, trainer, pl_module):
#         # Get the logged metrics
#         metrics = trainer.callback_metrics

#         # Format the 'Val_loss' value
#         if 'val_loss' in metrics:
#             metrics['Val_loss'] = '{:.1e}'.format(metrics['val_loss'])

#         # Format the 'Train_loss' value
#         if 'train_loss' in metrics:
#             metrics['Train_loss'] = '{:.1e}'.format(metrics['train_loss'])

def create_symlink(file_name, symlink_path):
    # Ensure paths are absolute
    # file_name = os.path.abspath(file_name)
    # symlink_path = os.path.abspath(symlink_path)

    try:
        # Attempt to create a symlink if it doesn't exist
        os.symlink(file_name, symlink_path, target_is_directory=False)
    except FileExistsError:
        # Handle the case where the symlink already exists
        try:
            os.remove(symlink_path)
            os.symlink(file_name, symlink_path, target_is_directory=False)
        except Exception as e:
            # Handle other potential exceptions (e.g., permission errors)
            print(f"Error creating symlink: {e}")
    except Exception as e:
        # Handle other potential exceptions (e.g., file not found)
        print(f"Error: {e}")