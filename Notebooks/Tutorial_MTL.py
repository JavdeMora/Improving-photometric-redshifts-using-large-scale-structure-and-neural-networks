# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: env_ai
#     language: python
#     name: env_ai
# ---

# %%
# # TUTORIAL TO MAKE A REDSHIFT PREDICTION WITH A NEURAL NETWORK 
# ### Import repo scripts
#Import photoz class from Photoz_method script
import sys
sys.path.append('../MTL')
from MTL_method import Photoz_MTL

# %%
# ### Define the model to train and make photo-z predictions

#Initialize the class
Model = Photoz_MTL(
    pathfile_photometry,
    pathfile_distances,
    pathfile_drand,
    photoz_hlayers = 5, 
    photoz_num_gauss = 5,
    epochs = 50,
    lr = 2e-3,
    lr_dpred=1e-5,
    batch_size = 100,
    min_sep = 0.03,
    max_sep = 26,
    nedges = 8
)

# %%
#Train the model with the data provided
Model.training(
    test_size=0.2,
    val_size=0.25 
)

# %%
#Make redshift prediction with color input
Model.pred_photoz(
    inputs_pathfile,
    all_rows=True,
    bands=['i', 'g', 'r', 'z', 'h', 'j', 'y'],
    plot=True
)

# %%
