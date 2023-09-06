# DIG: Draping Implicit Garment over the Human Body
<p align="center"><img src="misc/front.png"></p>

This is the repo for [**DIG: Draping Implicit Garment over the Human Body**](https://liren2515.github.io/page/dig/dig.html).

Here we provide the code and the pretrained models to generate garments and drape them on the posed human body.

## Pre-requisites
The code is implemented with python 3.6 and torch 1.9.0+cu102 (other versions may also work).

Download the female SMPL model from http://smplify.is.tue.mpg.de/ and place `basicModel_f_lbs_10_207_0_v1.0.0.pkl` in the folder of `./smpl_pytorch`.

Other dependencies include `trimesh`, `torchgeometry`, `scikit-image`.

### Docker support

In case you prefer Docker containers, build a Docker image using:

```
cd docker/
docker build -t $USER-dig .
cd ..
```

and run the container by:

```
# change <local-path-to-dig-repo>

docker run -it --rm --gpus all --shm-size=8gb --name $USER-dig -v <local-path-to-dig-repo>/:/DIG/ $USER-dig
```

### Troubleshoot

In case of `Subtraction, the '-' operator, with a bool tensor is not supported. If you are trying to invert a mask, use the '~' or 'logical_not()' operator instead.` issue with torchgeometry package, change the source file in `torchgeometry/core/conversions.py`:

```
# replace lines 301-308 with:

inv_mask_d0_d1 = ~mask_d0_d1
inv_mask_d0_nd1 = ~mask_d0_nd1
inv_mask_d2 = ~mask_d2
mask_c0 = mask_d2 * mask_d0_d1
mask_c1 = mask_d2 * inv_mask_d0_d1
mask_c2 = inv_mask_d2 * mask_d0_nd1
mask_c3 = inv_mask_d2 * inv_mask_d0_nd1
mask_c0 = mask_c0.view(-1, 1).type_as(q0)
mask_c1 = mask_c1.view(-1, 1).type_as(q1)
mask_c2 = mask_c2.view(-1, 1).type_as(q2)
mask_c3 = mask_c3.view(-1, 1).type_as(q3)
```

or check the original solution in the [comment](https://github.com/gaocong13/Projective-Spatial-Transformers/issues/3#issuecomment-718995585).

## To run
In `infer.py` you can find the script for inference. It loads the pretrained SDF models and skinning models (which are placed in `./extra-data/pretrained`), reconstructs garments using the learned latent codes and deforms them with the given SMPL parameters `/extra-data/pose-beta-sample.pt`. You can simply run the following command and the output mesh will be saved at `./output`.
```
python infer.py
```

To generate results for other garments and bodies, you can replace the latent code and SMPL parameters with other values.

Check [here](https://github.com/liren2515/DIG/tree/main/models) for the instruction of training.

## Citation
If you find this work helpful for your research, please cite
```
@inproceedings{ren2022dig,
  author = {Ren, Li and Guillard, Benoit and Remelli, Edoardo and Fua, Pascal},
  title = {{DIG: Draping Implicit Garment over the Human Body}},
  booktitle = {Asian Conference on Computer Vision},
  year = {2022}
}
```
