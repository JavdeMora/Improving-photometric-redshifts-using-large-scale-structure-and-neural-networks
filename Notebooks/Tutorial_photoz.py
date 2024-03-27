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
    photoz_hlayers = 5,
    photoz_num_gauss = 5, 
    epochs =2, 
    lr=1e-3, 
    batch_size=100, 
    pathfile='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/catalogues/FS2.csv'  
)


# %%
#Train the model with the data provided
Model.train_photoz(
    test_size=0.2,
    val_size=0.25 
)

# %%
#Make redshift prediction with color input
Model.pred_photoz(
    #Input the array of colors
    test_colors=[ 0.5304,  0.4784, -0.0526,  0.2745,  0.6954, -0.1624], 
    plot=True 
)

# %%
#Make redshift prediction for a set of colors
Model.pred_photoz_arr(
    #Input the array of colors
    test_colors = [
        [ 0.5304,  0.4784, -0.0526,  0.2745,  0.6954, -0.1624],
        [ 1.0710,  0.4091,  0.3013,  0.3987,  0.3970,  0.2368],
        [ 0.0593,  0.0383,  0.0917, -0.1004,  0.0179, -0.1460]
    ] ,
    plot=True
)

# %%
