from typing import Union
import torch
from abc import abstractmethod, ABC
from utils import utilities as ut
import numpy as np

class ObservationScheme():
    def __init__(self, samples):
        self.samples = samples
        return

    def pseudoobs_from_simulation(self):
        """
        make pseudoobs from an existing sequence of system states
        Returns:

        """
        pass

    @abstractmethod
    def define_operator(self, simulation):
        """

        Args:
            simulation: dict of tensors with axes (batch, time, ...)

        Returns:
            Observation operator object
        """
        pass

    def sample_pseudoobs(self, simulation):
        """This method is used to generate pseudo observation data from simulation. 

        Args:
            simulation (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(simulation, (list, torch.Tensor)):  # multiple simulations
            return [self.sample_pseudoobs(s) for s in simulation]
        obsop = self.define_operator(simulation)
        pseudoobs = obsop.observe(simulation)
        return pseudoobs, obsop
     

class BaseObservationOperator(ABC):
    def __init__(self,
                 sampler=None,
                 params=None,
                 data=None,
                 t_axis=None,
                 **kwargs):
        """

        Args:
            sampler:
            params:
            data:
            t_axis: a list of time points where we observe data
        """
        self.sampler = sampler
        self.params = params
        self.data = data
        self.t_axis = t_axis

    @abstractmethod
    def observe(self, simulation):
        pass

    @abstractmethod
    def nll(self, state, obs, t):
        pass


class MaskedGaussianNoise(BaseObservationOperator):
    # TODO: think of seeding mask and noise generation
    def __init__(self,
                 params=None,
                 p_obs: Union[float, dict]=0.5,  # scalar value or dict
                 noise_sd=1.0, # scalar value or dict
                 mask=None,
                 gridsize: dict[str:torch.Size]=None,
                 t_axis=None,
                 FixedObservationCount=False,
                 device=None,
                 **kwargs):
        """This is the method used to initialize an object

        Args:
            params (dict, optional): a dictionary containing info of parameters used. Defaults to None.
            p_obs (float, optional): probability distribution of data location of observational data. Defaults to 0.5.
            noise_sd (float, optinal): standard deviation of noise used in Gaussian distribution. Defafults to 1.0
            mask (dict, optional): a dictionary condtaining info of mask for each state, e.g., u (zonal velocity), v(meridional velocity), H (Pathymetry), etc.
            gridsize (dict, optional): a dictionary containing info of gridsize. Defaults to None.
            t_axis (_type_, optional): a list of time points where the data are observed. Defaults to None.
            FixedObservationCount (bool, optional): If it is set to True, the locations of observational data are fixed. Defaults to False.

        Raises:
            NotImplementedError: _description_
        """
        super(MaskedGaussianNoise).__init__()
        self.gridsize = gridsize

        if mask is not None:
            for k in mask.keys():
                assert k in gridsize.keys()
        else:
            mask = dict()
            if gridsize is not None:
                for k in gridsize.keys():
                    # print(k)
                    if not FixedObservationCount:
                        raise NotImplementedError
                    else:
                        # torch.manual_seed(idx)
                        # print(gridsize[k])
                        # print(torch.rand(1, len(t_axis), gridsize[k]))
                        mask[k] = (np.random.rand(*gridsize[k]) < p_obs[k]).astype(np.float32)

        self.mask = mask
        self.params = dict()
        self.params['noise_sd'] = noise_sd # TODO: per-field or not?
        self.t_axis = t_axis
        self.device = device

    def observe(self, simulation):
        """This is a place holder for the method that would generate observation data from simulation data

        Args:
            simulation (dict): a dictionary containing info of simulation data

        Returns:
            dict: a dictionary containing info of observation data
            e.g., in Tsunami {u: u_obs, u_mask: u_obs_mask, v: v_obs, v_mask: v_obs_mask, H: H_obs, H_mask: H_obs_mask}
        """
        
        
        # state = {'u': u, 'v': v, 'H': ssh}
        # state = torch.cat((ssh,u,v), dim=1)
        # we need to deal with t_time channel too
        
        # obs = dict()
        # for i, k in enumerate(self.gridsize.keys()):
        #     assert simulation[k].shape[0] == len(self.t_axis), f"Size mismatch between simulation of {k} and t_axis"
        #     obs[k] = simulation[k] + torch.randn_like(simulation[k]) * self.params['noise_sd']
        #     obs[k] *= self.mask[k]
        # return obs

        obs = np.empty_like(simulation)
        # # This code for 2D data
        # for i, k in enumerate(self.gridsize.keys()):
        #     assert simulation[:, i:i+1, :, :].shape[0] == len(self.t_axis), f"Size mismatch between simulation of {k} and t_axis"
        #     # obs[:, i:i+1, :, :] = simulation[:, i:i+1, :, :] + np.random.rand(*simulation[:, i:i+1, :, :].shape) * self.params['noise_sd']
        #     obs[:, i:i+1, :, :] = simulation[:, i:i+1, :, :] + np.random.randn(*simulation[:, i:i+1, :, :].shape) * self.params['noise_sd']
        #     # obs[k] *= self.mask[k].to(obs[k].dtype)
        #     obs[:, i:i+1, :, :] *= self.mask[k]
        #     # obs[k + '_mask'] = self.mask[k]
        # return obs
        # This code for 1D data
        for i, k in enumerate(self.gridsize.keys()):
            assert simulation[:, i:i+1, ...].shape[0] == len(self.t_axis), f"Size mismatch between simulation of {k} and t_axis"
            # obs[:, i:i+1, :, :] = simulation[:, i:i+1, :, :] + np.random.rand(*simulation[:, i:i+1, :, :].shape) * self.params['noise_sd']
            obs[:, i:i+1, ...] = simulation[:, i:i+1, ...] + np.random.randn(*simulation[:, i:i+1, ...].shape) * self.params['noise_sd']
            # obs[k] *= self.mask[k].to(obs[k].dtype)
            obs[:, i:i+1, ...] *= self.mask[k]
            # obs[k + '_mask'] = self.mask[k]
        return obs

    def nll(self, state, obs, time_idx):
        """This is a place holder for the method that would calculate negative log likelihood
            NLL = -logL = 0.5 * log(2 * pi) + log(sigma) + 0.5 * ((x - mu) / sigma)^2
            where mu is the mean (in this case, the state), sigma is the standard deviation (in this case, noise_sd), 
            and x is the observed value
            The valuation is done for only 1 data point.
        Args:
            state (dict): a dictionary containing info of state data
            obs (dict): a dictionary containing info of observation data
            t (_type_): _description_

        Returns:
            float: negative log likelihood
        """

        # nll = 0.0
        # for k in range(3): # 3 is 3 channels of data: ssh, u, and v
        #     logL = ((obs[:, k, :, :] - state[:, k, :, :]) / self.params['noise_sd'])**2
        #     nll -= logL.sum()
        # return nll

        mask = ut.dict2tsr(self.mask, dim=1)
        logL = ((obs - state*mask[time_idx]) / self.params['noise_sd'])**2
        nll = -logL.sum()
        return nll

        # for k in state.keys():
        #     logL = ((obs[k] - state[k]) / self.params['noise_sd'])**2
        #     nll -= logL.sum()
        # return nll

    def mseloss(self, state, obs, time_list):
        """This is a place holder for the method that would calculate mean square error loss
            MSE = 1/n * sum((x - mu)^2)
            where mu is the mean (in this case, the state), and x is the observed value
            The valuation is done for only 1 data point.
        Args:
            state (dict): a dictionary containing info of state data
            obs (dict): a dictionary containing info of observation data
            t (_type_): _description_
        Returns:
            float: mean square error loss
        """
        mask  = ut.dict2tsr(self.mask, dim=1)

        if isinstance(mask, np.ndarray):
            mask  = torch.from_numpy(mask).to(self.device)
        elif isinstance(mask, torch.Tensor):
            mask = mask.to(self.device)

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        elif isinstance(obs, torch.Tensor):
            obs = obs.to(self.device)

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)

        # mse   = ((obs - state * mask) ** 2).mean()
        mse   = ((obs - state * mask) ** 2).mean()

        return mse
    
        # mse = 0.0
        # for i, time in enumerate(time_list):
        #     # mask = ut.dict2tsr(self.mask, dim=1)
        #     mse += ((obs[i] - state[i]*mask[i])**2).mean()
        # return mse
    
        # mask = ut.dict2tsr(self.mask, dim=1)
        # mse = ((obs - state*mask[time_idx])**2).sum()/torch.numel(obs)
        # return mse
    

if __name__ == "__main__":
    from eip import util as ut
    import diff_ocean_nemo as donemo
    # xarray dataset 
    ds = ut.generate_data_from_datafolder(folder_path='/home/minh/coding/dev/machinelearning/diff_ocean_nemo/data/TSUNAMI-z-cD',
                                 file_name='TSUNAMI',
                                 number_of_files=1)
    ssh = ds['ssh'].values[0:1, 0:1]
    u = ds['baro_u'].values[0:1, 0:1]
    v = ds['baro_v'].values[0:1, 0:1]
