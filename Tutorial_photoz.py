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
#Import clustering class from clustering_method script
sys.path.append('photoz_method.py')
from photoz_method import photoz

# %%
#Initialize the class providing required and optional argumentsaaaaa
Model = photoz(
    #Required arguments (to pick by user)
    photoz_hlayers = 5, #Number of hidden layers of the network
    photoz_num_gauss = 5, #Number of output Gaussians
    epochs =2, #Number of epochs for the training
    
    #Set by default arguments (if left blank)
    lr=1e-3, #Learning rate for network training
    batch_size=100, #Batch size for network training
    pathfile='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/catalogues/FS2.csv'  #Path to file with fluxes catalog
)
    

# %%
#Train the model with the data provided
Model.train_photoz(
    
    #Set by default arguments (if left blank)
    test_size=0.2, #Subset size used for testing
    val_size=0.25 #Subset size used for training validation
    
)

# %%
#Make redshift prediction with color input
Model.pred_photoz(
    
    #Required arguments (to pick by user)
    test_colors=[ 0.5304,  0.4784, -0.0526,  0.2745,  0.6954, -0.1624], #Input the array of colors
    
    #Set by default arguments (if left blank)
    plot=True #Choose if you want to plot the redshift probability distribution
    
)

# %%
Model.pred_photoz_arr(
    
    #Required arguments (to pick by user)
    test_colors = [
        [ 0.5304,  0.4784, -0.0526,  0.2745,  0.6954, -0.1624],
        [ 1.0710,  0.4091,  0.3013,  0.3987,  0.3970,  0.2368],
        [ 0.0593,  0.0383,  0.0917, -0.1004,  0.0179, -0.1460]
    ] , #Input the arrays of colors (more than 1)
    
    #Set by default arguments (if left blank)
    plot=True #Choose if you want to plot the redshift probability distribution for each galaxy
)

# %%
