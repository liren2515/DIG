# About Training

Here are the scripts for the training of our SDF and skinning models. In `model_sdf.py` and `model_skinning.py`, you can find how the loss terms are implemented and what values we use for weight balancing (check the function `optimize_parameters()`). 

Note that the training of our skinnig model depends on `kaolin-0.1.0`, since we use it to convert the body mesh to an SDF for the purpose of collision penalization.
