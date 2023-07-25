import glob
import os

import torch
import argparse
import numpy as np
import random
from pathlib import Path
import os
import imageio
import yaml
import io
import medutils
import json

import wandb
from utils.basic import parse_config, import_module
from torch.utils.data import DataLoader

from utils.vis import angle2color, k2img, alpha2img
from utils.eval import bias_corr, get_eval_metrics


def main():
    # parse args and get config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_abdominal_iconik.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-r', '--r_acc', type=int, default=None)
    parser.add_argument('-sub', '--subject', type=str, default=None)
    parser.add_argument('-s', '--slice', type=int, default=None)
    parser.add_argument('-log', '--log', type=str, default='wandb')

    parser.add_argument('-seed', '--seed', type=int, default=0)
    args = parser.parse_args()

    # enable Double precision
    torch.set_default_dtype(torch.float32)

    # set gpu and random seed
    # torch.cuda.set_device(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # parse config
    config = parse_config(args.config)
    config['data_root'] = os.path.join(Path.home(), config["data_root"])
    config["results_root"] = os.path.join(Path.home(), config["results_root"])

    # config['slice_name'] = slice_name
    config['gpu'] = args.gpu
    config['exp_summary'] = args.log

    # optional from command line (otherwise in config)
    if args.r_acc is not None:
        config['acc_factor'] = args.r_acc
    if args.subject is not None:
        config['subject_name'] = args.subject
    if args.slice is not None:
        config['slice'] = args.slice

    # create dataset
    dataset_class = import_module(config["dataset"]["module"], config["dataset"]["name"])
    dataset = dataset_class(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'],
                            drop_last = True)
    config['nc'] = dataset.n_coils

    # create model
    model_class = import_module(config["model"]["module"], config["model"]["name"])
    NIKmodel = model_class(config)
    NIKmodel.init_train()

    if config['exp_summary'] == 'wandb':
        # log params
        params_to_log = []
        for idx, (name, param) in enumerate(NIKmodel.named_parameters()):
            if "omega" in name:
                params_to_log.append(name)
        # wandb.watch(NIKmodel, params_to_log, log_graph=True)
        # wandb.watch(NIKmodel, "all", log_graph=False)

    loaded_epoch = 0

    ### Load pretrained MLP
    if config['pretrain']:

        # pretrain_path = os.path.join(config["results_root"], config["pretrain_path"], '_e' + str(config["pretrain_epoch"]))
        pretrain_path = os.path.join(config["results_root"], config["pretrain_group"] +
                                     "_S" + str(config["subject_name"]),
                                     config["pretrain_exp"] + "_slice" + str(config["slice"]) +
                                     "_R" + str(config["acc_factor"]))
        # find latest model
        pretrain_run_id = os.listdir(pretrain_path)[0]

        pretrained_model_path = os.path.join(pretrain_path, pretrain_run_id,
                                             "model_checkpoints", "best_model")

        # load B for features
        model_dict = NIKmodel.network_kdata.state_dict()
        pretrained_dict = torch.load(pretrained_model_path, map_location=NIKmodel.device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        NIKmodel.network_kdata.load_state_dict(model_dict)

        # load SirenNet weights
        model_dict = NIKmodel.network_kdata.siren_net.state_dict()
        pretrained_dict = torch.load(pretrained_model_path, map_location=NIKmodel.device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        NIKmodel.network_kdata.siren_net.load_state_dict(model_dict)

        ## update kernel weights if existant
        if "diff_loss" in NIKmodel.config["model"]["params"] and NIKmodel.config["model"]["params"]["diff_loss"]:
            if "pretrained_kernel" in NIKmodel.config["patch_schedule"]:
                pretrained_with_kernel_dict = torch.load(os.path.join(config["results_root"],
                                                                      NIKmodel.config["patch_schedule"]["pretrained_kernel"],
                                                                      "best_model"),
                                                                      map_location=NIKmodel.device)
                NIKmodel.network_kdata.load_state_dict(pretrained_with_kernel_dict)
                loaded_epoch = 20

    # save config with all changed params for later evaluation
    with io.open(NIKmodel.model_save_path + '/config.yml', 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

    loss_model = 1e10

    for epoch in range(loaded_epoch, config['num_steps']):

        if 'patch_schedule' in config:
            if config['patch_schedule']['calib_timing'] == "None":
                pass
            elif config['patch_schedule']['calib_timing'] == "jump":
                if epoch <= config['patch_schedule']['freeze_epoch']:

                    ## calibrate kernel
                    dataloader.dataset.increment = config['patch_schedule']['calib_region']

                    ## freeze MLP and train CONV in calib region
                    for param in NIKmodel.network_kdata.siren_net.parameters():
                        param.requires_grad = False

                    if hasattr(NIKmodel.network_kdata, 'conv_patch'):
                        for param in NIKmodel.network_kdata.conv_patch.parameters():
                            param.requires_grad = True
                    if hasattr(NIKmodel.network_kdata, 'conv_patch_t'):
                        for param in NIKmodel.network_kdata.conv_patch_t.parameters():
                            param.requires_grad = True
                        for param in NIKmodel.network_kdata.conv_patch_x.parameters():
                            param.requires_grad = True

                    loss_epoch = 0
                    for i, sample in enumerate(dataloader):
                        # kcoord, kv = sample['coords'], sample['target']
                        loss = NIKmodel.train_batch(sample)
                        print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}")
                        loss_epoch += loss

                else:

                    ## apply kernel
                    # Load complete region and freeze CONV
                    dataloader.dataset.increment = 1.0
                    for param in NIKmodel.network_kdata.siren_net.parameters():
                        param.requires_grad = True
                    for param in NIKmodel.network_kdata.conv_patch.parameters():
                        param.requires_grad = False

                    loss_epoch = 0
                    for i, sample in enumerate(dataloader):
                        # kcoord, kv = sample['coords'], sample['target']
                        if "diff_loss" in NIKmodel.config["model"]["params"] and NIKmodel.config["model"]["params"]["diff_loss"]:
                            loss = NIKmodel.train_batch(sample, diff_loss = True)
                        else:
                            loss = NIKmodel.train_batch(sample)
                        print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}")
                        loss_epoch += loss


        log_dict = {
            'epoch': epoch,
            'loss': loss_epoch / len(dataloader)}

        # log test reconstruction for each step
        if 'log_test' in config and config["log_test"]:
            kpred = NIKmodel.test_batch()

            # if the k-space was cropped in FE, it was rescaled to -1 to 1 for training -
            # this needs to be reversed before the FFT
            if "fe_crop" in config and config["fe_crop"] < 1:
                crop_factor = int(1 / config["fe_crop"])
                n_fe_crop = int(kpred.shape[-1] * config["fe_crop"]) # assumes same sampling in x & y
                crop_start = int((kpred.shape[-1]- n_fe_crop)/2)
                kpred_filled = torch.zeros_like(kpred)
                kpred_filled[:,:,crop_start:crop_start+n_fe_crop,crop_start:crop_start+n_fe_crop] = kpred[:,:,::crop_factor,
                                                                                                    ::crop_factor]
                kpred = kpred_filled

            # kpred = kpred[:,:, config['slice'], ...]
            # vis_img = k2img(kpred.unsqueeze(1), csm = csm)
            vis_img = k2img(kpred, csm = dataset.csm)
            # vis_img = k2img(kpred.unsqueeze(0), csm = csm)

            log_dict.update({
                'k': wandb.Video(vis_img['k_mag'], fps=1, format="gif"),
                'img': wandb.Video(vis_img['combined_mag'], fps=1, format="gif"),
                'img_phase': wandb.Video(vis_img['combined_phase'], fps=1, format="gif"),
                'khist': wandb.Histogram(torch.view_as_real(kpred).detach().cpu().numpy().flatten()),
            })


            # log intermediate k-space
            kpred_intermediate = NIKmodel.test_batch(conv=False)
            vis_img_intermediate = k2img(kpred_intermediate, csm = dataset.csm)
            log_dict.update({
                'k_interm': wandb.Video(vis_img_intermediate['k_mag'], fps=1, format="gif"),
                'img_interm': wandb.Video(vis_img_intermediate['combined_mag'], fps=1, format="gif"),
                'img_phase_interm': wandb.Video(vis_img_intermediate['combined_phase'], fps=1, format="gif"),
                'khist_interm': wandb.Histogram(torch.view_as_real(kpred_intermediate).detach().cpu().numpy().flatten()),
            })

            if config['exp_summary'] == 'wandb':
                if hasattr(NIKmodel, 'network_phase'):
                    alpha = NIKmodel.test_batch_phase()
                    phase_img = alpha2img(alpha)
                    log_dict["alpha"] = wandb.Video(phase_img['alpha'], fps=10, format="gif")
                    log_dict['alpha_color'] = wandb.Video(phase_img['alpha_color'], fps=10, format="gif")
                if hasattr(NIKmodel, 'conv_patch'):
                    # weights = NIKmodel.conv_patch.weight.detach()
                    conv_mag = torch.abs(NIKmodel.conv_patch.weight).detach().cpu().numpy()
                    conv_phase = torch.angle(NIKmodel.conv_patch.weight).detach().cpu().numpy()

                    conv_mag_vid = np.stack([medutils.visualization.plot_array(conv_mag[i, ...]) for i in range(conv_mag.shape[0])])
                    conv_phase_vid = np.stack([medutils.visualization.plot_array(conv_phase[i,...])for i in range(conv_phase.shape[0])])

                    # conv_phase_col = angle2color(conv_phase, vmin=-np.pi, vmax=np.pi)
                    # conv_real = torch.real(weights)
                    # conv_imag = torch.imag(weights)

                    log_dict["conv_patch_mag_vid"] = wandb.Video(conv_mag_vid[:, None, ...], fps=10, format="gif")
                    log_dict["conv_patch_phase"] = wandb.Video(conv_phase_vid[:,None,...], fps=10, format="gif")
                    # log_dict["conv_patch_mag"] = wandb.Image(conv_mag)
                    # log_dict["conv_patch_phase"] = wandb.Video(conv_phase_col, fps=10, format="gif")

        # log progress
        NIKmodel.exp_summary_log(log_dict)

        # save checkpoints
        if loss_model > loss_epoch:
            l = loss_epoch/len(dataloader)
            NIKmodel.save_best_network(epoch, l.detach().item())
            loss_model = loss_epoch
            # save best images

        if epoch % 50 == 0 or epoch == 20:
            NIKmodel.save_network("_e{}".format(epoch))
            # save images

            # middle index
            if hasattr(NIKmodel, 'log_test') and config["log_test"]:
                t = int(np.floor(vis_img["combined_mag"].shape[0] / 2))
                imageio.imwrite(NIKmodel.model_save_path + '/recon_middlnav_e{}.png'.format(epoch),
                                vis_img["combined_mag"][t, ...].squeeze())

    ## save final reconstruction
    kpred = NIKmodel.test_batch()
    vis_img = k2img(kpred, csm = dataset.csm, scale = False)
    vis_img_comp = k2img(kpred, csm = dataset.csm, scale = True)

    kpred_intermediate = NIKmodel.test_batch(conv=False)
    vis_img_intermediate = k2img(kpred, csm=dataset.csm, scale=False)

    results_path = os.path.join(os.path.dirname(NIKmodel.model_save_path), 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.savez(results_path + '/{}.npz'.format(vis_img["combined_mag"].shape),
             img=vis_img["combined_mag"],
             img_c=vis_img["combined_img"])
    np.savez(results_path + '/{}_intermediate.npz'.format(vis_img["combined_mag"].shape),
             img=vis_img_intermediate["combined_mag"],
             img_c=vis_img_intermediate["combined_img"])


    ### save best model
    best_model_info = NIKmodel.load_best_network()
    kpred = NIKmodel.test_batch()
    vis_img = k2img(kpred, csm = dataset.csm, scale = False)
    vis_img_comp = k2img(kpred, csm = dataset.csm, scale = True)
    results_path = os.path.join(os.path.dirname(NIKmodel.model_save_path), 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.savez(results_path + '/{}_best_model.npz'.format(vis_img["combined_mag"].shape),
             img=vis_img["combined_mag"],
             img_c = vis_img["combined_img"])

    ## compute eval metrics
    # ToDo: requires path to reference
    # ref_path = os.path.join(Path.home(),
    #                         "workspace_results/shared/grasprecon/S{}/recons/sl{}".format(config['subject_name'],
    #                                                                                      config['slice']))
    # ref = glob.glob(ref_path + "/*R1*.npz", recursive=True)[0]
    # ref_recon = np.load(ref)["ref"].squeeze()
    #
    # # correct bias
    # pred_corr = np.stack([bias_corr(vis_img["combined_img"][ms,0,...], ref_recon[ms,...])
    #                       for ms in range(ref_recon.shape[0])])
    # # get eval metrics
    # best_model_info.update({"ref_path": ref})
    # best_model_info.update(get_eval_metrics(pred_corr[0, ...], ref_recon[0, ...]))

    if config['exp_summary'] == 'wandb':
        wandb.log(best_model_info)

    with open(results_path + '_best_model.txt', 'w') as f:
        json.dump(best_model_info,f)



if __name__ == '__main__':
    main()