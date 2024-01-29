import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.cm as cm
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
#from Photo_z_architecture import photoz_network

class MTL_photoz:
    def _init_(self):
        self.net_photoz = net_photoz()

    def get_loader_fluxes(self, filetype, test_size, val_size, batch_size, data_dir='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/catalogues/FS2.csv',*args):
        #Transform raw data
        if filetype == 'csv':
            parquet = pd.read_csv(str(data_dir),sep =',', header=0, comment='#')
        elif filetype == 'parquet':
            parquet = pd.read_parquet(str(data_dir),sep =',', header=0, comment='#')
        else: 
            raise ValueError("Only filetype =='csv' and 'parquet' are supported")

        parquet_labeled=parquet.rename(columns={'euclid_vis_el_model3_ext_odonnell_ext':'vis','euclid_vis_el_model3_ext_odonnell_ext_error_realization':'err_vis','lsst_g_el_model3_ext_odonnell_ext':'g','lsst_i_el_model3_ext_odonnell_ext':'i','lsst_r_el_model3_ext_odonnell_ext':'r','lsst_z_el_model3_ext_odonnell_ext':'z','euclid_nisp_y_el_model3_ext_odonnell_ext':'y','euclid_nisp_j_el_model3_ext_odonnell_ext':'j','euclid_nisp_h_el_model3_ext':'h','lsst_g_el_model3_ext_odonnell_ext_error_realization':'err_g','lsst_i_el_model3_ext_odonnell_ext_error_realization':'err_i','lsst_r_el_model3_ext_odonnell_ext_error_realization':'err_r','lsst_z_el_model3_ext_odonnell_ext_error_realization':'err_z','euclid_nisp_y_el_model3_ext_odonnell_ext_error_realization':'err_y','euclid_nisp_j_el_model3_ext_odonnell_ext_error_realization':'err_j','euclid_nisp_h_el_model3_ext_odonnell_ext_error_realization':'err_h',})
        #Add noise
        parquet_labeled['i']=parquet_labeled['i']+parquet_labeled['err_i']
        parquet_labeled['g']=parquet_labeled['g']+parquet_labeled['err_g']
        parquet_labeled['r']=parquet_labeled['r']+parquet_labeled['err_r']
        parquet_labeled['z']=parquet_labeled['z']+parquet_labeled['err_z']
        parquet_labeled['h']=parquet_labeled['h']+parquet_labeled['err_h']
        parquet_labeled['j']=parquet_labeled['j']+parquet_labeled['err_j']
        parquet_labeled['y']=parquet_labeled['y']+parquet_labeled['err_y']
        #Mags
        parquet_labeled['i']= -2.5*np.log10(parquet_labeled['i'])-48.6
        parquet_labeled['g']= -2.5*np.log10(parquet_labeled['g'])-48.6
        parquet_labeled['r']= -2.5*np.log10(parquet_labeled['r'])-48.6
        parquet_labeled['z']= -2.5*np.log10(parquet_labeled['z'])-48.6
        parquet_labeled['h']= -2.5*np.log10(parquet_labeled['h'])-48.6
        parquet_labeled['j']= -2.5*np.log10(parquet_labeled['j'])-48.6
        parquet_labeled['y']= -2.5*np.log10(parquet_labeled['y'])-48.6
        parquet_labeled['vis']= -2.5*np.log10(parquet_labeled['vis'])-48.6
        #Drop NaN and noise columns
        parquet_labeled=parquet_labeled.dropna(axis=0,how='any')
        parquet_labeled=parquet_labeled.drop(['err_z','err_i','err_y','err_j','err_h','err_g','err_r','err_vis','dec_gal','ra_gal'], axis=1)
        #Filter data (Mag_i < 25, z<1)
        filtered_parquet0=parquet_labeled[parquet_labeled['i']< 25]
        filtered_parquet=filtered_parquet0[filtered_parquet0['observed_redshift_gal']<1]
        dataset = filtered_parquet
        #Create colour dataframe
        colors_df = pd.DataFrame(np.c_[dataset['observed_redshift_gal'],dataset['vis'],dataset['g']-dataset['r'],dataset['r']-dataset['i'],dataset['i']-dataset['z'], dataset['z']-dataset['y'], dataset['y']-dataset['j'],dataset['j']-dataset['h']], columns=['observed_redshift_gal','Mag_i','g-r','r-i','i-z','z-y','y-j','j-h'])
        #Split data to train, validation and test datasets
        test_size= test_size
        train_dataset, test_dataset=train_test_split(colors_df, test_size=test_size)
        #Split data to train, validation and test datasets
        val_size = val_size
        train_df, val_df = train_test_split(train_dataset, test_size=val_size)
        #Create Tensor datasets
        input_labs = ['g-r','r-i','i-z','z-y','y-j','j-h']
        target_lab = ['observed_redshift_gal']
        train_input=torch.Tensor(train_df[input_labs].values)
        val_input= torch.Tensor(val_df[input_labs].values)
        train_target = torch.Tensor(train_df[target_lab].values)
        val_target = torch.Tensor(val_df[target_lab].values)
        i_test=test_dataset['Mag_i'].values
        test_input=torch.Tensor(test_dataset[input_labs].values)
        test_target=test_dataset[target_lab].values
        train_dataset = TensorDataset(train_input,train_target)
        val_dataset = TensorDataset(val_input,val_target)
        #Create loaders
        batch_size=batch_size
        loader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        loader_val = DataLoader(val_dataset, batch_size = batch_size, shuffle =False)
        
        return loader_train, loader_val
    
  def get_training_distances(self,*args):
  
  def train_photoz(self,training_data):
  training_data=self.get_training_fluxes
  z = self.net_photoz(training_data)
  def train_clustering(self,*args):
  training_data=self.get_training_distances
  def train_mtl(self, *args):im
