import torch
import torch.nn as nn
import numpy as np
import os

from utils.mri import coilcombine, ifft2c_mri
from .base import NIKBase
from models.siren_sos_nik import NIKSiren
from utils.basic import import_module

from merlinth.layers.convolutional.complex_conv import ComplexConv2d
from merlinth.layers.complex_act import cReLU

class NIKSirenPatch(NIKSiren):
    def __init__(self, config) -> None:
        super().__init__(config)

        # B = torch.randn((self.config["coord_dim"], self.config["feature_dim"] // 2), dtype=torch.float32)
        # self.register_buffer('B', B)
        # self.conv_patch = ComplexConv2d(self.config["out_dim"], self.config["out_dim"],
        #                                 kernel_size=self.patch_dim, bias=self.config["patch_bias"])

        self.create_network()
        self.to(self.device)

    def create_network(self):
        # overwrite outdim with number of channels
        self.config["out_dim"] = self.config["nc"]
        out_dim = self.config["out_dim"]
        coord_dim = self.config["coord_dim"]
        self.patch_dim =  self.config["model"]["params"]["patch_dim"]

        self.network_kdata = ICoSiren(coord_dim, out_dim,
                                   **self.config['model']['params']).to(self.device) # feature_dim, num_layers, omega_0, omega_scale
        # self.network_kdata = Siren(coord_dim, feature_dim, num_layers, out_dim,
        #                           omega_0=omega).to(self.device)

    def create_optimizer(self):
        """Create the optimizer."""
        # self.optimizer = torch.optim.Adam([self.parameters(), self.network.parameters()], lr=self.config['lr'])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr']))

        ## create second optimizer for subnetwork
        if "dual_loss" in self.config["model"]["params"] and self.config["model"]["params"]["dual_loss"]:
            self.optimizer_intermediate = torch.optim.Adam(self.network_kdata.siren_net.parameters(), lr=float(self.config['lr']))
        # for param in self.named_parameters():
        #     print(param[0])

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


    def init_test(self):
        """Initialize the network for testing.
        Should be called before testing.
        It does the following things:f
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

    def pre_process(self, inputs,  conv = True):
        """ker
        Preprocess the input coordinates.
        """

        # if self.network_kdata.training:

        if conv:

            patch_dist = self.config["model"]["params"]["patch_dist"] if "patch_dist" in self.config["model"][
                "params"] else 1.0

            # define spacing between neighboring points
            dx = patch_dist * 2.0 / self.config['nx']
            dy = patch_dist * 2.0 / self.config['ny']
            dpatch = int(np.floor(self.patch_dim / 2))

            # create kernel
            kernel = torch.stack(torch.meshgrid(torch.linspace(-dpatch, dpatch, 2 * dpatch + 1),
                                    torch.linspace(-dpatch, dpatch, 2 * dpatch + 1),
                                    indexing="ij"), axis = -1)
            kernel[..., 0] = kernel[..., 0] * dx
            kernel[..., 1] = kernel[..., 1] * dy
            kernel_flat = np.reshape(kernel, (-1, 2))

            # inputs['coords'] = inputs['coords'].to(self.device)
            inputs['coords_patch'] = np.repeat(inputs['coords'].unsqueeze(1), kernel_flat.shape[0],  axis=1)
            inputs['coords_patch'][..., 1:3] =  inputs['coords_patch'][..., 1:3].clone() + kernel_flat     # add offset to x any y coord
            inputs['coords_patch'] = inputs['coords_patch'].reshape(-1, inputs['coords'].shape[-1])
            inputs['coords_patch'] = inputs['coords_patch'].to(self.device)

            # features = self.network_kdata.pre_process(inputs['coords_patch'])
            features = torch.cat([torch.sin(inputs['coords_patch'] @ self.network_kdata.B),
                                  torch.cos(inputs['coords_patch'] @ self.network_kdata.B)], dim=-1)

            inputs['coords'] = inputs['coords'].to(self.device)  # required for loss calculation

        else:
            inputs['coords'] = inputs['coords'].to(self.device)  # required for loss calculation#
            features = torch.cat([torch.sin(inputs['coords'] @ self.network_kdata.B),
                                  torch.cos(inputs['coords'] @ self.network_kdata.B)], dim=-1)



        inputs['features'] = features

        if inputs.keys().__contains__('targets'):
            inputs['targets'] = inputs['targets'].to(self.device)

        return inputs

    def post_process(self, output, conv=True, intermed_output = False):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        # B, samples * patchdim
        # output = torch.complex(output[..., 0:self.config["out_dim"]], output[..., self.config["out_dim"]:])

        if conv:
            # if self.network_kdata.training:
            # merge samples back to patches
            output = torch.reshape(output, (-1, self.patch_dim, self.patch_dim, self.config["out_dim"]*2)) # B, C, patchdim, patchdim
            output = torch.complex(output[..., 0:self.config["out_dim"]], output[..., self.config["out_dim"]:])
            output = output.permute(0,3,2,1)

            # Dual loss requires the intermediate predicted value of the network as well as the convolved value
            if self.network_kdata.training and "dual_loss" in self.config["model"]["params"] and self.config["model"]["params"]["dual_loss"] or intermed_output:
                output_intermediate = output[:,:, int(self.patch_dim/2), int(self.patch_dim/2)].clone()

            if "mask_center" in self.config["model"]["params"] and self.config["model"]["params"]["mask_center"]:
                output[:,:, int(self.patch_dim/2), int(self.patch_dim/2)] *= 0

            # apply kernel
            output = self.network_kdata.conv_patch(output.clone())
            output = output.squeeze(-1).squeeze(-1)

            if self.network_kdata.training and "dual_loss" in self.config["model"]["params"] and \
                    self.config["model"]["params"]["dual_loss"] or intermed_output:
                return output, output_intermediate
            else:
                return output

        else:
            # do not apply convolutional layer
            output = torch.complex(output[..., 0:self.config["out_dim"]], output[..., self.config["out_dim"]:])
            return output


    def train_batch(self, sample):

        torch.autograd.set_detect_anomaly(True)

        self.network_kdata.train()
        self.network_kdata.conv_patch.train()

        self.optimizer.zero_grad()
        if "dual_loss" in self.config["model"]["params"] and self.config["model"]["params"]["dual_loss"]:
            self.optimizer_intermediate.zero_grad()


        sample = self.pre_process(sample)
        output = self.forward(sample)

        ## Calculate loss depending on config setting
        if "dual_loss" in self.config["model"]["params"] and self.config["model"]["params"]["dual_loss"]:
            output_conv, output_intermediate = self.post_process(output)
            loss1, reg2 = self.criterion(output_intermediate, sample['targets'], sample['coords'])
            loss2, reg = self.criterion(output_conv, sample['targets'], sample['coords'])

            log_loss_dict = {
                'loss1': loss1,
                'loss_intermediate': loss2}
            self.exp_summary_log(log_loss_dict)

            # print("Loss1: ", loss1)
            # print("Loss2: ", loss2)

            # separated optim
            # loss1.backward(retain_graph=True)
            # loss2.backward(retain_graph=False)
            # self.optimizer_intermediate.step()
            # self.optimizer.step()

            # joint optim
            loss = loss1 + loss2

            loss.backward()
            self.optimizer.step()

        elif "diff_loss" in self.config["model"]["params"] and self.config["model"]["params"]["diff_loss"]:
            output_conv, output_intermediate = self.post_process(output)

            loss = torch.abs(output_conv - output_intermediate)
            loss.backward()
            self.optimizer.step()
            # loss1, reg2 = self.criterion(output_intermediate, sample['targets'], sample['coords'])
            # loss2, reg = self.criterion(output_conv, sample['targets'], sample['coords'])


        else:
            # loss calculated solely for output of NIK or after ICo layer
            output = self.post_process(output)
            loss, reg = self.criterion(output, sample['targets'], sample['coords'])
            loss.backward()
            self.optimizer.step()

        return loss

    def test_batch(self, input=None, conv = True):
        """
        Test the network with a cartesian grid.
        if sample is not None, it will return image combined with coil sensitivity.
        """
        self.network_kdata.eval()
        # self.conv_patch.eval() if hasattr(self, 'conv_patch') else None

        with torch.no_grad():

            if input is None:
                nc = self.config['nc']
                nx = self.config['nx']
                ny = self.config['ny']
                nnav = self.config['nnav']

                kxs = torch.linspace(-1, 1 - 2 / nx, nx)
                kys = torch.linspace(-1, 1 - 2 / ny, ny)
                # knav = torch.linspace(self.config["dataset"]["navigator_min"], 1, nnav)
                knav = torch.linspace(-1 + 1 / nnav, 1 - 1 / nnav, nnav)

                # TODO: discard the outside coordinates before prediction
                grid_coords = torch.stack(torch.meshgrid(knav, kys, kxs, indexing='ij'), -1)  #  nav, nx, ny,3
                # grid_coords = grid_coords.to(self.device)
                dist_to_center = torch.sqrt(grid_coords[..., 1] ** 2 + grid_coords[..., 2] ** 2)
                dist_to_center = dist_to_center.unsqueeze(1).expand(-1, nc, -1, -1)  # nt, nc, nx, ny

                nDim = grid_coords.shape[-1]
                contr_split = 1

            else:
                grid_coords = input
                # grid_coords = grid_coords.to(self.device)
                nDim = grid_coords.shape[-1]
                contr_split = 1

            # split t for memory saving
            contr_split_num = np.ceil(grid_coords.shape[0] / contr_split).astype(int)

            kpred_list = []
            for t_batch in range(contr_split_num):
                grid_coords_batch = grid_coords[t_batch * contr_split:(t_batch + 1) * contr_split]

                grid_coords_batch = grid_coords_batch.reshape(-1, nDim).requires_grad_(False)
                # get prediction
                sample = {'coords': grid_coords_batch}
                sample = self.pre_process(sample, conv=conv)  # encode time differently?
                kpred = self.forward(sample)
                kpred = self.post_process(kpred, conv=conv)
                kpred_list.append(kpred)
            kpred = torch.concat(kpred_list, 0)

            if input is None:
                # TODO: clearning this part of code
                kpred = kpred.reshape(nnav, ny, nx, nc).permute(0,3,1,2) # coil dimension second, imgDim last
                k_outer = 1
                kpred[dist_to_center >= k_outer] = 0
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

class ICoSiren(nn.Module):
    def __init__(self, coord_dim, out_dim, hidden_features, num_layers, patch_dim, patch_layers, patch_bias, omega_0=30, exp_out=True,
                 relu=True, complex_conv = True, **kwargs) -> None:
        super().__init__()

        B = torch.randn((coord_dim, hidden_features // 2), dtype=torch.float32)
        self.register_buffer('B', B)

        self.conv_patch = []


        ## Create conv kernel based on model params
        if complex_conv:
            for i in range(patch_layers - 1):
                self.conv_patch.append(ComplexConv2d(out_dim, out_dim, kernel_size=patch_dim, bias=patch_bias, padding="same"))
                if relu:
                    self.conv_patch.append(cReLU())
            self.conv_patch.append(ComplexConv2d(out_dim, out_dim, kernel_size=patch_dim, bias=patch_bias, padding="valid"))
        else:
            for i in range(patch_layers - 1):
                self.conv_patch.append(nn.Conv2d(out_dim, out_dim, kernel_size=patch_dim, bias=patch_bias, padding="same"))
                if relu:
                    self.conv_patch.append(nn.ReLU())
            self.conv_patch.append(nn.Conv2d(out_dim, out_dim, kernel_size=patch_dim, bias=patch_bias, padding="valid"))

        self.conv_patch = nn.Sequential(*self.conv_patch)

        self.siren_net = Siren(coord_dim, hidden_features, num_layers, out_dim,
                                  omega_0=omega_0)

    def forward(self, features):
        return self.siren_net(features)

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
