# ICoNIK: Informed Correction of Neural Implicit Representation of K-Space


This is the repo for the paper _ICoNIK: Generating Respiratory-Resolved Abdominal MR Reconstructions Using Neural Implicit Representations in k-Space_ 
published at the DGM4MICCAI Workshop 2023.


## Content
- `train_sos_nik`: First neural implicit k-space representation (NIK) for motion-resolved abdominal reconstruction of
 radial Stack-of-Star (SoS) data guided by a respiratory navigator signal (extension of [NIK_MRI](https://github.com/wenqihuang/NIK_MRI))
- `train_sos_iconik`: Informed correction of NIK (ICoNIK) to leverage neighborhood information by applying a kernel which is auto-calibrated on a more densely sampled region.
## Data
The expected source data is a numpy file with the following entries:
~~~
    kdata: raw k-space data - np.array of shape (ech, coils, z-slices, spokes x FE_steps)  
    self_nav: respiratory navgiator signal - np.array of shape (spokes,)
    traj: sample trajectory - np.array of shape (ech, spokes x FE_steps, 2)
    csm: coil sensitivity maps - np.array of shape (coild, x, y, z)
~~~

## Training

1) For NIK on radial Stack-of-Star (SoS) data
 
   a) Adapt `config_abdominal_nik.yml` to your needs (e.g. for data, experiment naming & tracking, model, loss and recon)

   b) Run `python3 -u train_sos_nik.py -c "configs/config_abdominal_nik.yml" -r $ACC_FACTOR -sub $SUBJECT -s $SLICE`


2) For ICoNIK on radial Stack-of-Star (SoS) data:

   a) Adapt `config_abdominal_iconik.yml` to your needs:
      - Set path to pretrained NIK in training config
      - Set ICo parameters (ACS calibration schedule: kernel-size, autocalibration region and freeze_epoch)
      - General config, model config, loss config, recon config: as in 1)

   b) Run `python3 -u train_sos_iconik.py -c "configs/config_abdominal_iconik.yml" -r $ACC_FACTOR -sub $SUBJECT -s $SLICE`


All computations were performed using Python 3.10.1 and PyTorch 1.13.1.

# If you find this code useful, please cite:
For generative respiratory-resolved reconstruction using NIK and parallel imaging-inspired ICoNIK:

    @misc{TBD}

For neural implicit k-space representation learning:

     @inproceedings{Huang_2023,
      abstract = author = {Huang, Wenqi and Li, Hongwei Bran and Pan, Jiazhen and Cruz, Gastao and Rueckert, Daniel and Hammernik, Kerstin},
      title = {Neural Implicit k-Space for Binning-Free Non-Cartesian Cardiac MR Imaging},
      pages = {548--560},
      publisher = {{Springer Nature Switzerland}},
      isbn = {978-3-031-34048-2},
      editor = {Frangi, Alejandro and de Bruijne, Marleen and Wassermann, Demian and Navab, Nassir},
      booktitle = {Information Processing in Medical Imaging},
      year = {2023},
     }
