import torch
import torch.nn as nn
import numpy as np
import os, datetime

from utils.mri import coilcombine, ifft2c_mri
from .base import NIKBase
from utils.basic import import_module


class NIKSoSBase(NIKBase):
    def __init__(self, config) -> None:
        """
        Base model for Radial Stack-of-Stars applications
        """
        super().__init__(config)
        self.create_network()
        self.to(self.device)

    def load_names(self):
        """
        Generate and set experiment and group names, and paths for model checkpoints and results.

        This method sets the experiment and group IDs based on the provided configuration.
        It creates the experiment path using the current date and time.
        The model_save_path and results_save_path are generated based on the results_root in the configuration.

        Note:
            Ensure that the configuration (self.config) contains the necessary parameters for generating the paths.
        """

        exp_id = f"_hdr{self.config['hdr_ff_factor']}_slice{self.config['slice']}_R{self.config['acc_factor']}"
        group_id = f'S{self.config["subject_name"]}'

        if "group_name" in self.config:
            group_id = "_".join([self.config["group_name"], group_id])
        if "exp_name" in self.config:
            exp_id = "_".join([self.config["exp_name"], exp_id])

        exp_path = os.path.join(group_id, exp_id, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if "results_root" in self.config:
            model_save_path = "".join([self.config["results_root"], '/', exp_path, '/model_checkpoints'])
            results_save_path = "".join([self.config["results_root"], '/', exp_path, '/results'])

        self.exp_id = exp_id
        self.group_id = group_id
        self.model_save_path = model_save_path
        self.results_save_path = results_save_path

    def create_network(self):
        """
        Create the neural network for the specified model.

        This method creates the neural network using the provided configuration parameters.
        It sets self.network_kdata as the Siren network with the specified dimensions.

        Note:
            The configuration (self.config) must contain the necessary parameters for network creation.
            Adjust the import and instantiation code for your custom Siren class based on your module structure.
        """

        self.config["out_dim"] = self.config["nc"]
        out_dim = self.config["out_dim"]
        coord_dim = self.config["coord_dim"]

        self.network_kdata = Siren(coord_dim, out_dim,
                                  **self.config['model']['params']).to(self.device)

    def init_expsummary(self):
        """
        Initialize the visualization tools.
        Should be called in init_train after the initialization of self.exp_id.
        """
        if self.config['exp_summary'] == 'wandb':
            import wandb
            self.exp_summary = wandb.init(
                project=self.config['wandb_project'],
                name=self.exp_id,
                config=self.config,
                group=self.group_id,
                entity=self.config['wandb_entity']
            )

    def init_train(self):
        """Initialize the network for training.
        Should be called before training.
        It does the following things:
            1. set the network to train mode
            2. create the optimizer to self.optimizer
            3. create the model save directory
            4. initialize the visualization tools
        If you want to add more things, you can override this function.
        """
        self.network_kdata.train()

        self.create_criterion()
        self.create_optimizer()

        self.load_names()

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.init_expsummary()

    def init_test(self):
        """Initialize the network for testing.
        Should be called before testing.
        It does the following things:
            1. set the network to eval mode
            2. load the network parameters from the weight file path
        If you want to add more things, you can override this function.
        """
        self.weight_path = self.config['weight_path']

        self.load_network()

        self.network_kdata.eval()

        exp_id = self.weight_path.split('/')[-2]
        epoch_id = self.weight_path.split('/')[-1].split('.')[0]
        # TODO: add exp and epoch id to the result save path when needed

        # setup model save dir
        results_save_dir = os.path.join(self.group_id, self.exp_id)
        if "results_root" in self.config:
            results_save_dir = "".join([self.config["results_root"],'/', results_save_dir])

        if not os.path.exists(results_save_dir):
            os.makedirs(results_save_dir)

        self.result_save_path


    def save_network(self, name):
        """Save the network parameters to the path."""
        path = os.path.join(self.model_save_path, name)
        torch.save(self.network_kdata.state_dict(), path)

    def load_network(self):
        """Load the network parameters from the path."""
        path = self.config['weight_path']
        self.network_kdata.load_state_dict(torch.load(path, map_location=self.device))

    def load_best_network(self):
        """Load the network parameters from the path."""
        path = os.path.join(self.model_save_path, "best_model")
        self.network_kdata.load_state_dict(torch.load(path, map_location=self.device))
        with open(path + '.txt', 'r') as f:
            d = dict([line.strip().split(':', 1) for line in f])
        return d

    def save_best_network(self, epoch, loss):
        """Save the network parameters to the path."""
        path = os.path.join(self.model_save_path, "best_model")
        torch.save(self.network_kdata.state_dict(), path)

        with open(path + '.txt', 'w') as f:
            f.write('epoch: {} \n'.format(epoch))
            f.write('loss: {}'.format(loss))

    def pre_process(self, inputs):
        """
        Preprocess the input coordinates.
        """
        inputs['coords'] = inputs['coords'].to(self.device)

        features = self.network_kdata.pre_process(inputs['coords'])
        inputs['features'] = features

        if inputs.keys().__contains__('targets'):
            inputs['targets'] = inputs['targets'].to(self.device)

        return inputs

    def post_process(self, output):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output = torch.complex(output[..., 0:self.config["out_dim"]], output[..., self.config["out_dim"]:])
        return output

    def train_batch(self, sample):
        self.optimizer.zero_grad()
        sample = self.pre_process(sample)
        output = self.forward(sample)
        output = self.post_process(output)
        loss, reg = self.criterion(output, sample['targets'], sample['coords'])
        loss.backward()
        self.optimizer.step()
        return loss

    def test_batch(self, input=None):
        """
        Test the network with a cartesian grid.
        if sample is not None, it will return image combined with coil sensitivity.
        """
        with torch.no_grad():

            if input is None:
                # ncontr = self.config['ncontr']
                nc = self.config['nc'] #len(self.config['coil_select'])  # nc = self.config['nc']
                nx = self.config['nx']
                ny = self.config['ny']
                nnav = self.config['nnav']

                # coordinates: contr, kx, ky, nc, nav
                # contrs = torch.linspace(-1, 1, ncontr)
                kc = torch.linspace(-1, 1, nc)
                kxs = torch.linspace(-1, 1 - 2 / nx, nx)
                kys = torch.linspace(-1, 1 - 2 / ny, ny)
                knav = torch.linspace(self.config["dataset"]["navigator_min"], 1, nnav)

                # TODO: disgard the outside coordinates before prediction
                grid_coords = torch.stack(torch.meshgrid(kc, kys, kxs, knav, indexing='ij'), -1).to(
                    self.device)  # ncontr, nc, nx, ny, nav, 5
                dist_to_center = torch.sqrt(grid_coords[..., 1] ** 2 + grid_coords[..., 2] ** 2)
                nDim = grid_coords.shape[-1]
                contr_split = 2

            else:
                grid_coords = input.to(self.device)
                nDim = grid_coords.shape[-1]
                contr_split = 2

            # split t for memory saving
            contr_split_num = np.ceil(grid_coords.shape[0] / contr_split).astype(int)

            kpred_list = []
            for t_batch in range(contr_split_num):
                grid_coords_batch = grid_coords[t_batch * contr_split:(t_batch + 1) * contr_split]

                grid_coords_batch = grid_coords_batch.reshape(-1, nDim).requires_grad_(False)
                # get prediction
                sample = {'coords': grid_coords_batch}
                sample = self.pre_process(sample)  # encode time differently?
                kpred = self.forward(sample)
                # kpred *= canonical frame
                # alpha = conditionalMoCo.forward(x,y,t)
                # kpred * alpha

                kpred = self.post_process(kpred)
                kpred_list.append(kpred)
            kpred = torch.concat(kpred_list, 0)

            # kpred_list.append(kpred)
            # kpred = torch.mean(torch.stack(kpred_list, 0), 0) #* filter_value.reshape(-1, 1)

            if input is None:
                # TODO: clearning this part of code
                kpred = kpred.reshape(nc, ny, nx, nnav)
                k_outer = 1
                kpred[dist_to_center >= k_outer] = 0
                kpred = kpred.permute(3, 0, 1, 2)  # coil dimension second, imgDim last
                # kpred = kpred.squeeze(-1)
            return kpred

    def forward(self, inputs):
        return self.network_kdata(inputs['features'])


"""
The following code is a demo of mlp with sine activation function.
We suggest to only use the mlp model class to do the very specific 
mlp task: takes a feature vector and outputs a vector. The encoding 
and post-process of the input coordinates and output should be done 
outside of the mlp model (e.g. in the prepocess and postprocess 
function in your NIK model class).
"""

class Siren(nn.Module):
    def __init__(self, coord_dim, hidden_features, num_layers, out_dim, omega_0=30, exp_out=True) -> None:
        super().__init__()

        self.net = [SineLayer(hidden_features, hidden_features, is_first=True, omega_0=omega_0)]
        for i in range(num_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        final_linear = nn.Linear(hidden_features, out_dim * 2)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0,
                                         np.sqrt(6 / hidden_features) / omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

        B = torch.randn((coord_dim, hidden_features // 2), dtype=torch.float32)
        self.register_buffer('B', B)

    def pre_process(self, input):
        input = torch.cat([torch.sin(input @ self.B),
                           torch.cos(input @ self.B)], dim=-1)
        return input

    def forward(self, features):
        return self.net(features)


class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
