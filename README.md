# ICoNIK: Informed Correction of Neural Implicit Representation of k-Space


This is the repo for the paper _ICoNIK: Generating Respiratory-Resolved Abdominal MR Reconstructions Using Neural Implicit Representations in k-Space_ 
accepted at the ![DGM4MICCAI Workshop 2023](https://dgm4miccai.github.io/) | ![PrePrint](https://arxiv.org/abs/2308.08830)

**Abstract:**
_Motion-resolved reconstruction for abdominal magnetic resonance imaging (MRI) remains a challenge due to the trade-off between residual motion blurring caused by discretized motion states and undersampling artefacts. In this work, we propose to generate blurring-free motion-resolved abdominal reconstructions by learning a neural implicit representation directly in k-space (NIK). Using measured sampling points and a data-derived respiratory navigator signal, we train a network to generate continuous signal values. To aid the regularization of sparsely sampled regions, we introduce an additional informed correction layer (ICo), which leverages information from neighboring regions to correct NIK's prediction. Our proposed generative reconstruction methods, NIK and ICoNIK, outperform standard motion-resolved reconstruction techniques and provide a promising solution to address motion artefacts in abdominal MRI._

![General Overview](/overview.pdf?raw=true "General Overview")


## Content
- `train_sos_nik`: First neural implicit k-space representation (NIK) for motion-resolved abdominal reconstruction of
 radial Stack-of-Star (SoS) data guided by a respiratory navigator signal (extension of [NIK_MRI](https://github.com/wenqihuang/NIK_MRI))
- `train_sos_iconik`: Informed correction of NIK (ICoNIK) to leverage neighborhood information by applying a kernel which is auto-calibrated on a more densely sampled region.

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

    @article{Spieker_2023_ICoNIK,
     author = {Spieker, Veronika and Huang, Wenqi and Eichhorn, Hannah and Stelter, Jonathan and Weiss, Kilian and Zimmer, Veronika A. and Braren, Rickmer F. and Karampinos, Dimitrios C. and Hammernik, Kerstin and Schnabel, Julia A.},
     title = {{ICoNIK}: Generating Respiratory-Resolved Abdominal {MR} Reconstructions Using Neural Implicit Representations in k-Space},
     publisher = {{Springer Nature Switzerland}},
     booktitle={Deep Generative Models. {DGM4MICCAI} [in press]},
     year = {2023},
     note = {arXiv: 2308.08830}
    }

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
