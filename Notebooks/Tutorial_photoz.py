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
#     display_name: insight
#     language: python
#     name: insight
# ---

# %% [markdown]
# # TUTORIAL TO MAKE A REDSHIFT PREDICTION WITH A NEURAL NETWORK

# %% [markdown]
# ### Import repo scripts

# %%
#Import photoz class from Photoz_method script
import sys
sys.path.append('../mtl_lss')
from Photoz_method import photoz

# %% [markdown]
# ### Define the model to train and make photo-z predictions

# %%
#Initialize the class
Model = photoz(
    pathfile,  
    photoz_hlayers = 5,
    photoz_num_gauss = 5, 
    epochs =2, 
    lr=1e-3, 
    batch_size=100
)


# %%
#Train the model with the data provided
Model.train_photoz(
    test_size=0.2,
    val_size=0.25 
)

# %% [markdown]
# ## ANOTHER THING: THE COLORS ARE NOT 'PICKED BY THE USER'. YOU SHOULD LOAD A GALAXY FROM THE FILE AND TEST IT THERE

# %%
#Make redshift prediction with color input
Model.pred_photoz(
    inputs_pathfile,
    all_rows=True,
    bands=['i', 'g', 'r', 'z', 'h', 'j', 'y'],
    plot=True
)
