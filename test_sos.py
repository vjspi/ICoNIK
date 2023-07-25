import glob
import os

import torch
import argparse
import numpy as np
import random
from pathlib import Path
import os
import imageio
import medutils
import matplotlib.pyplot as plt
import sys
import json

import wandb

import utils.vis
import utils.eval
from datasets.radialSoS_sorted import RadialSoSDataset
from utils.basic import parse_config, import_module
from torch.utils.data import DataLoader

from utils.vis import angle2color, k2img, alpha2img

def main(path, nt = 20):


    print("NT:", nt)
    # parse args and get config
    parser = argparse.ArgumentParser()

    # parser.add_argument('-p', '--path', type=str, default='configs/config_abdominal.yml')
    # parser.add_argument('-g', '--gpu', type=int, default=0)
    # parser.add_argument('-r', '--r_acc', type=int, default=None)
    # parser.add_argument('-sub', '--subject', type=str, default=None)
    # parser.add_argument('-s', '--slice', type=int, default=None)

    # parser.add_argument('-s', '--seed', type=int, default=0)
    # args = parser.parse_args()

    # enable Double precision
    torch.set_default_dtype(torch.float32)

    ## Load config file from model checkpoint folder
    # parse config
    # ToDo: Insert path
    model_path = os.path.join(Path.home(),
                               "path/to/",
                                path)
    results_path = os.path.join(model_path,
                                "rec_test")
    weights_path = os.path.join(model_path,
                                "model_checkpoints")

    for i in os.listdir(weights_path):
        if os.path.isfile(os.path.join(weights_path, i)) and 'config' in i:
            config = parse_config(os.path.join(weights_path,i))

    config['data_root'] = os.path.join(Path.home(), config["data_root"][config["data_root"].find("workspace"):]) # need to cut home directory from training
    config["results_root"] = os.path.join(Path.home(), config["results_root"][config["results_root"].find("workspace"):]) # need to cut home directory from training
    config['gpu'] = 0


    ## Select model weights
    # model = "_e400"
    model = "best_model"
    config["weight_path"] = os.path.join(weights_path, model)
    ## Load information about model
    with open(config["weight_path"] + '.txt', 'r') as f:
        best_model_info = dict([line.strip().split(':', 1) for line in f])

    # config["weight_path"] = os.path.join(weights_path, "best_model")

    # optional from command line (otherwise in config)
    # if args.r_acc is not None:
    #     config['acc_factor'] = args.r_acc
    # if args.subject is not None:
    #     config['subject_name'] = args.subject
    # if args.slice is not None:
    #     config['slice'] = args.slice

    # create model
    model_class = import_module(config["model"]["module"], config["model"]["name"])
    NIKmodel = model_class(config)
    NIKmodel.load_names()
    NIKmodel.init_test()

    file_name = f"{config['data_root']}S{config['subject_name']}/test_reconSoS.npz"
    data = np.load(file_name)
    csm = data["smaps"][..., config["slice"]]


    ### Define the desired points to be sampled
    if NIKmodel.config["coord_dim"] == 4:
        ksamples = [26, 300, 300, nt] # kc, kx, ky, nav
        kc = torch.linspace(-1, 1, ksamples[0])
        kxs = torch.linspace(-1, 1 - 2 / ksamples[1], ksamples[1])
        kys = torch.linspace(-1, 1 - 2 / ksamples[2], ksamples[2])
        # knav = torch.linspace(-1, 1, ksamples[3])
        knav = torch.linspace(-1 + 1 / ksamples[3], 1 - 1 / ksamples[3], ksamples[3])

        grid_coords = torch.stack(torch.meshgrid(kc, kxs, kys, knav, indexing='ij'), -1)  # nt, nx, ny, nc, 4
        dist_to_center = torch.sqrt(grid_coords[..., 1] ** 2 + grid_coords[..., 2] ** 2)

        # Run model
        kpred = NIKmodel.test_batch(input=grid_coords)

        kpred = kpred.reshape(*ksamples)
        k_outer = 1
        kpred[dist_to_center >= k_outer] = 0
        kpred = kpred.permute(3, 0, 1, 2)

    elif NIKmodel.config["coord_dim"] == 3:
        ksamples = [nt, 300, 300]  # kx, ky, nav
        # kc = torch.linspace(-1, 1, ksamples[0])
        kxs = torch.linspace(-1, 1 - 2 / ksamples[1], ksamples[1])
        kys = torch.linspace(-1, 1 - 2 / ksamples[2], ksamples[2])
        # knav = torch.linspace(-1, 1, ksamples[0])
        knav = torch.linspace(-1 + 1 /  ksamples[0], 1 - 1 / ksamples[0],  ksamples[0])

        grid_coords = torch.stack(torch.meshgrid(knav, kys, kxs, indexing='ij'), -1)  # nav, nx, ny,3
        dist_to_center = torch.sqrt(grid_coords[..., 1] ** 2 + grid_coords[..., 2] ** 2)
        dist_to_center = dist_to_center.unsqueeze(1).expand(-1, NIKmodel.config["nc"], -1, -1)
        # Run model
        kpred = NIKmodel.test_batch(input=grid_coords)

        kpred = kpred.reshape(*ksamples, NIKmodel.config["nc"]).permute(0,3,1,2)
        k_outer = 1
        kpred[dist_to_center >= k_outer] = 0

    assert len(ksamples) == NIKmodel.config["coord_dim"]

    # kpred = kpred[:,:, config['slice'], ...]
    # vis_img = k2img(kpred.unsqueeze(1), csm = csm)

    ### save best model
    vis_img = k2img(kpred, csm = csm, scale = False)
    vis_img_comp = k2img(kpred, csm = csm, scale = True)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    np.savez(results_path + '/{}_best_model.npz'.format(ksamples),
             img=vis_img["combined_mag"],
             img_c=vis_img["combined_img"])

    utils.vis.save_gif(vis_img_comp["combined_mag"],
                       duration= 4000,
                       filename= results_path +'/dyn{}_sub{}_sl{}_r{}.gif'.format(kpred.shape[0], config['subject_name'],config['slice'], config['acc_factor']))

    utils.vis.save_gif(vis_img_comp["combined_mag"],
                       duration= 4000,
                       filename= results_path +'/dyn{}_sub{}_sl{}_r{}_int2.gif'.format(kpred.shape[0], config['subject_name'],config['slice'], config['acc_factor']),
                       intensity_factor=2.0)


    for t in range(kpred.shape[0]):
        imageio.imwrite(results_path + '/nav{}_{}.png'.format(knav[t], model),
                        vis_img["combined_mag"][t,...].squeeze())


    ## compute eval metrics
    ### Load reference and compare
    ref_path = os.path.join(Path.home(),
                            "workspace_results/shared/grasprecon/S{}/recons/sl{}".
                            format(config['subject_name'],config['slice']))
    ref = glob.glob(ref_path + "/*R1*.npz", recursive=True)[0]
    ref_recon = np.load(ref)["ref"].squeeze()

    # correct bias
    # pred_corr = np.stack([utils.eval.bias_corr(vis_img["combined_img"][ms,0,...], ref_recon[ms,...])
    #                       for ms in range(ref_recon.shape[0])])
    #
    ms = ref_recon.shape[0]
    nt_idx=int(np.ceil(0.5 * nt / ms))
    pred_corr = utils.eval.bias_corr(vis_img["combined_img"][nt_idx,...].squeeze(),
                                     ref_recon[0,...])

    # get eval metrics
    best_model_info.update({"ref_path": ref})
    # best_model_info.update(utils.eval.get_eval_metrics(pred_corr[0, ...], ref_recon[0, ...]))
    best_model_info.update(utils.eval.get_eval_metrics(pred_corr, ref_recon[0, ...]))

    with open(results_path + '_best_model.txt', 'w') as f:
        json.dump(best_model_info,f)

    # plot_normalized(pred_corr)
    # plot_normalized(vis_img["combined_img"][:,0,...])

    print("Done")


def plot_normalized(array):
    fig, ax = plt.subplots(array.shape[0], 1, figsize = (array.shape[0]*10, 10))

    for idx, i in enumerate(zip(range(array.shape[0]), ax.flatten())):
        img = medutils.visualization.normalize(array[idx,...])
        l = np.percentile(img, 99)
        ax[idx].imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=l)
        ax[idx].axis('off')

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='configs/config_abdominal.yml')
    parser.add_argument('-n', '--nt', type=int, default=20)
    args = parser.parse_args()

    main(args.path, args.nt)