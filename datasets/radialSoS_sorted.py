from torch.utils.data import Dataset
import torch
import numpy as np
import h5py


class RadialSoSDataset(Dataset):
    def __init__(self, config):
        """
        Custom PyTorch Dataset for handling radial SoS MRI data.

        Parameters:
            config (dict): Configuration dictionary containing necessary parameters.
        """
        super().__init__()
        self.config = config

        ## load data
        file_name = f"{config['data_root']}S{config['subject_name']}/test_reconSoS.npz"
        data = np.load(file_name)

        ## sort data
        kdata = data["kspace"]  # (2,26,84,360600) -> (ech, coils, slices, spokesxFE))
        self_nav = data["pcaCurve"]  # (601,) -> (spokes)
        ref = data["ref"]  # (x, y, slices, ech*dyn)
        traj = data["traj"]  # (2, 360600, 2) -> (ech, FExPE, kx/ky)
        # load slice specific sensitivity map
        self.csm = data["smaps"][..., config["slice"]]

        im_size = self.csm.shape
        print("Im_size shape", im_size)


        n_fe = self.config["fe_steps"]
        n_slices = self.config["n_slices"]
        n_coils = im_size[0]    # imsize is coils * x * y

        if n_coils != len(config["coil_select"]):
            print("Careful: coil selection does not match sensitivity maps")

        # normalize kdata
        kdata /= np.max(np.abs(kdata))

        ## reshape data
        kdata = kdata.reshape(2, n_coils, n_slices, -1, n_fe)  # (ech, coils, slices, spokes, FE)
        traj = traj.reshape(2, -1, n_fe, 2)
        print("Trajectory shape:", traj.shape)

        # %% Data prepocessing
        ### Acceleration (Phase encoding)
        if "acc_factor" in self.config:
            nspoke_acc = int(kdata.shape[-2] / self.config["acc_factor"])
            kdata = kdata[:, :, :, :nspoke_acc, :]
            traj = traj[:, :nspoke_acc, :, :]
            self_nav = self_nav[:nspoke_acc]

        if self.config["dataset"]["navigator_min"] == -1:
            self_nav = ((self_nav - (self_nav.max() + self_nav.min()) / 2) / (self_nav.max() - self_nav.min())) * 2     # normalize to -1 to 1
        elif self.config["dataset"]["navigator_min"] == 0:
            self_nav = (self_nav - self_nav.min()) / (self_nav.max() - self_nav.min())                                  # normalize to 0 to 1


        ### Downsamling (Frequency encoding)
        if "fe_downsample" in self.config and self.config["fe_downsample"] > 1:
            assert isinstance(self.config["fe_downsample"], int)
            n_fe_ds = int(n_fe / self.config["fe_downsample"])
            kdata = kdata[:,:,:,:,::self.config["fe_downsample"]]
            traj = traj[..., ::self.config["fe_downsample"],:]
            assert kdata.shape[-1] == n_fe_ds
            n_fe = n_fe_ds

        ### Reduce resolution by cropping the frequency range (CAREFUL: Still rescaled back for training!)
        if "fe_crop" in self.config and self.config["fe_crop"] < 1:
            n_fe_crop = int(n_fe * self.config["fe_crop"])
            crop_start = int((n_fe - n_fe_crop)/2)
            traj = traj[:,:,crop_start:crop_start+n_fe_crop,:]
            kdata = kdata[...,crop_start:crop_start+n_fe_crop]
            assert kdata.shape[-1] == n_fe_crop
            assert np.all(np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2) < self.config["fe_crop"])
            n_fe = n_fe_crop
            traj *= (1/self.config["fe_crop"])
            print("Warning: Trajectory FE direction got cropped and rescaled again - consider in final reconstruction voxel size ")

        if "calib_crop" in self.config and self.config["calib_crop"] < 1:
            n_fe_crop = int(n_fe * self.config["calib_crop"])
            crop_start = int((n_fe - n_fe_crop)/2)
            traj = traj[:,:,crop_start:crop_start+n_fe_crop,:]
            kdata = kdata[...,crop_start:crop_start+n_fe_crop]
            assert kdata.shape[-1] == n_fe_crop
            assert np.all(np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2) < self.config["calib_crop"])
            n_fe = n_fe_crop
            # traj *= (1/self.config["fe_crop"])
            print("Warning: Trajectory FE direction got cropped but NOT rescaled")

        assert kdata.shape[-2] == traj.shape[-3]
        n_spokes = kdata.shape[-2]

        ### crop data to desired echo/slice
        slice = self.config['slice']
        echo = 0
        kdata = kdata[echo, :, slice, :, :]  # minimize data
        traj = traj[echo, :]

        # kcoords = np.zeros((1, n_spokes, n_fe, 3))    # (nav, spokes, FE, 3) -> nav, ky, (contr, slices, spokes, FE, coils, 5) -> contr, kx, ky, nc, nav
        # contr = torch.linspace(-1, 1, n_contr)
        # kcoords[:, :, :, 0] = np.reshape(self_nav, (1, n_spokes, n_fe))
        # kcoords[:, :, :, 1] = np.reshape(traj[..., 0] * 2, (1, n_spokes, n_fe))
        # kcoords[:, :, :, 2] = np.reshape(traj[..., 1] * 2, (1, n_spokes, n_fe))
        # self.kcoords = np.reshape(kcoords.astype(np.float32), (-1, 3))

        ### move coil dimension to output
        nDim = self.config["coord_dim"]
        kcoords = np.zeros((n_spokes, n_fe, nDim))  # (nav, spokes, FE, 3) -> nav, ky, (contr, slices, spokes, FE, coils, 5) -> contr, kx, ky, nc, nav
        klatent = np.zeros((n_spokes, n_fe, 1))
        # kc = torch.linspace(-1, 1, n_coils)
        ## Save spoke number for each sample
        kspoke = np.zeros((n_spokes, n_fe, 1))
        # contr = torch.linspace(-1, 1, n_contr)

        # kcoords[:, :, :, :, 0] = np.reshape(self_nav, (1, n_spokes, n_fe))
        # kcoords[..., 0] = np.reshape(kc, (n_coils, 1, 1))
        kcoords[..., 1] = np.reshape(traj[..., 0] * 2, (1, n_spokes, n_fe))  # ky
        kcoords[..., 2] = np.reshape(traj[..., 1] * 2, (1, n_spokes, n_fe))  # kx
        kcoords[..., 0] = np.reshape(self_nav, (1, n_spokes, 1))
        klatent[..., 0] = np.reshape(self_nav, (1, n_spokes, 1))
        # kcoords[:, :, :, 3] = np.reshape(self_nav, (1, n_spokes, n_fe))
        kspoke[..., 0] = np.reshape(np.linspace(0, 1, n_spokes), (n_spokes, 1))

        ### put coils to output
        assert kdata.shape[0] == n_coils        # coils x spokes x FE
        kdata = np.transpose(kdata, (1,2,0))    # spokes x FE x coils


        # sort data from center to outer edge if calib region is required
        # ToDo: Clean condition
        if "patch_schedule" in self.config and self.config["patch_schedule"]["calib_region"] < 1:
            kcoords = np.reshape(kcoords.astype(np.float32), (-1, nDim))
            klatent = np.reshape(klatent.astype(np.float32), (-1, 1))
            kdata = np.reshape(kdata.astype(np.complex64), (-1, n_coils))
            kspoke = np.reshape(kspoke.astype(np.float32), (-1, 1))

            dist_to_center = np.sqrt(kcoords[..., 1] ** 2 + kcoords[..., 2] ** 2)
            idx = np.argsort(dist_to_center)

            self.kcoords = kcoords[idx].astype(np.float32)
            self.klatent = klatent[idx].astype(np.float32)
            self.kdata = kdata[idx].astype(np.complex64)  # (nav*spokes*FE, 1)
            self.kspoke = kspoke[idx].astype(np.float32)

        else:
            self.kcoords = np.reshape(kcoords.astype(np.float32), (-1, nDim))
            self.klatent = np.reshape(klatent.astype(np.float32), (-1, 1))
            self.kdata = np.reshape(kdata.astype(np.complex64), (-1, n_coils))  # (nav*spokes*FE, 1)
            self.kspoke = np.reshape(kspoke.astype(np.float32), (-1, 1))


        self.kcoords_flat = torch.from_numpy(self.kcoords)
        self.klatent_flat = torch.from_numpy(self.klatent)
        self.kdata_flat = torch.from_numpy(self.kdata)
        self.kspoke_flat = torch.from_numpy(self.kspoke)

        self.n_kpoints = self.kcoords.shape[0]
        self.n_coils = n_coils

        self.increment = 1  # 100 percent as default (all points are sampled)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.n_kpoints

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the specified index.

        Parameters:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'coords', 'latent', 'targets', and 'kspoke' tensors for the sample.
        """
        
        # In case the pool of samples is reduced (i.e. increment <1): Map the index to another the reduced range
        # mapped_index = index projected to desired increment region (e.g. with desired increment of 0.4 and 100 kpoints, index 60 corresponds to 20)
        no_samples = np.int(np.floor(self.increment * self.n_kpoints))
        index = index % no_samples      # maps all data points to specified range

        # point wise sampling
        sample = {
            'coords': self.kcoords_flat[index],
            'latent': self.klatent_flat[index],
            'targets': self.kdata_flat[index],
            'kspoke': self.kspoke_flat[index]
        }
        return sample
