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
import sys

sys.path.append('Photo_z_architecture.py')
from Photo_z_architecture import photoz_network



class MTL_photoz:
    def __init__(self, photoz_hlayers, photoz_num_gauss,epochs,lr=1e-3 ,batch_size = 100, pathfile='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/catalogues/FS2.csv'):
        self.net_photoz = photoz_network(photoz_hlayers, photoz_num_gauss).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        cat=self._get_colorsdf(pathfile=pathfile)
        self.cat=cat
        self.test_input=torch.Tensor(self.cat.loc[0][['g-r','r-i','i-z','z-y','y-j','j-h']].values)# esto es solo para testear
        
    def _get_colorsdf(self, filetype='csv', pathfile='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/catalogues/FS2.csv', df= 'None', *args):
        #Transform raw data
        if filetype == 'csv':
            parquet = pd.read_csv(str(pathfile),sep =',', header=0, comment='#')
        elif filetype == 'parquet':
            parquet = pd.read_parquet(str(pathfile),sep =',', header=0, comment='#')
        elif filetype == 'dataframe':
            parquet = df 
        else: 
            raise ValueError("Only filetype =='csv' and 'parquet' are supported")
            
        if 'i' in parquet.columns and 'g' in parquet.columns and 'r' in parquet.columns and 'z' in parquet.columns and 'h' in parquet.columns and 'j' in parquet.columns and 'y' in parquet.columns:
            parquet_labeled=parquet
        
        else:
            parquet_labeled=parquet.rename(columns={'euclid_vis_el_model3_ext_odonnell_ext':'vis','euclid_vis_el_model3_ext_odonnell_ext_error_realization':'err_vis','lsst_g_el_model3_ext_odonnell_ext':'g','lsst_i_el_model3_ext_odonnell_ext':'i','lsst_r_el_model3_ext_odonnell_ext':'r','lsst_z_el_model3_ext_odonnell_ext':'z','euclid_nisp_y_el_model3_ext_odonnell_ext':'y','euclid_nisp_j_el_model3_ext_odonnell_ext':'j','euclid_nisp_h_el_model3_ext':'h','lsst_g_el_model3_ext_odonnell_ext_error_realization':'err_g','lsst_i_el_model3_ext_odonnell_ext_error_realization':'err_i','lsst_r_el_model3_ext_odonnell_ext_error_realization':'err_r','lsst_z_el_model3_ext_odonnell_ext_error_realization':'err_z','euclid_nisp_y_el_model3_ext_odonnell_ext_error_realization':'err_y','euclid_nisp_j_el_model3_ext_odonnell_ext_error_realization':'err_j','euclid_nisp_h_el_model3_ext_odonnell_ext_error_realization':'err_h',})
            parquet_labeled['i']=parquet_labeled['i']+parquet_labeled['err_i']
            parquet_labeled['g']=parquet_labeled['g']+parquet_labeled['err_g']
            parquet_labeled['r']=parquet_labeled['r']+parquet_labeled['err_r']
            parquet_labeled['z']=parquet_labeled['z']+parquet_labeled['err_z']
            parquet_labeled['h']=parquet_labeled['h']+parquet_labeled['err_h']
            parquet_labeled['j']=parquet_labeled['j']+parquet_labeled['err_j']
            parquet_labeled['y']=parquet_labeled['y']+parquet_labeled['err_y']
            parquet_labeled=parquet_labeled.drop(['err_z','err_i','err_y','err_j','err_h','err_g','err_r','err_vis','dec_gal','ra_gal'], axis=1)
        #Mags
        parquet_labeled['i']= -2.5*np.log10(parquet_labeled['i'])-48.6
        parquet_labeled['g']= -2.5*np.log10(parquet_labeled['g'])-48.6
        parquet_labeled['r']= -2.5*np.log10(parquet_labeled['r'])-48.6
        parquet_labeled['z']= -2.5*np.log10(parquet_labeled['z'])-48.6
        parquet_labeled['h']= -2.5*np.log10(parquet_labeled['h'])-48.6
        parquet_labeled['j']= -2.5*np.log10(parquet_labeled['j'])-48.6
        parquet_labeled['y']= -2.5*np.log10(parquet_labeled['y'])-48.6
        if 'vis' in parquet_labeled.columns:
            parquet_labeled['vis']= -2.5*np.log10(parquet_labeled['vis'])-48.6
        #Drop NaN
        parquet_labeled=parquet_labeled.dropna(axis=0,how='any')
        
        #Filter data (Mag_i < 25, z<1)
        filtered_parquet=parquet_labeled[parquet_labeled['i']< 25]
        if 'observed_redshift_gal' in filtered_parquet.columns:
            filtered_parquet=filtered_parquet[filtered_parquet['observed_redshift_gal']<1]
        dataset = filtered_parquet
        #Create colour dataframe
        if 'vis' in parquet_labeled.columns and 'observed_redshift_gal' in parquet_labeled.columns:
            colors_df = pd.DataFrame(np.c_[dataset['observed_redshift_gal'],dataset['vis'],dataset['g']-dataset['r'],dataset['r']-dataset['i'],dataset['i']-dataset['z'], dataset['z']-dataset['y'], dataset['y']-dataset['j'],dataset['j']-dataset['h']], columns=['observed_redshift_gal','Mag_i','g-r','r-i','i-z','z-y','y-j','j-h'])
        else:
            colors_df = pd.DataFrame(np.c_[dataset['g']-dataset['r'],dataset['r']-dataset['i'],dataset['i']-dataset['z'], dataset['z']-dataset['y'], dataset['y']-dataset['j'],dataset['j']-dataset['h']], columns=['g-r','r-i','i-z','z-y','y-j','j-h'])
        return colors_df
        
    def _get_loaders(self, test_size, val_size, batch_size,  *args):
        #Split data to train, validation and test datasets
        test_size= test_size
        train_dataset, test_dataset=train_test_split(self.cat, test_size=test_size)
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

    def train_photoz(self, test_size=0.2, val_size=0.25, *args): #argumento solo catalogo
        loader_train, loader_val = self._get_loaders(test_size, val_size, self.batch_size)
        net =  self.net_photoz.cuda()
        train_losses = [] 
        alpha_list = []
        mu_list = []
        ztrue_list = []
        optimizer = optim.Adam(net.parameters(), lr=self.lr) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        for epoch in range(self.epochs):
            for datain, xeval in loader_train:
                optimizer.zero_grad() 
                logalpha, mu, logsig = net(datain.to(self.device))
                sig = torch.exp(logsig)

                log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((xeval.to(self.device)[:,None] - mu[:,:,None])/sig[:,:,None])**2
                log_prob = torch.logsumexp(log_prob, 1)
                loss = - log_prob.mean()

                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                train_losses.append(train_loss)

            scheduler.step()

            net.eval()
            val_losses = []
            logalpha_list = []
            out_pred, out_true = [],[]
            
            with torch.no_grad():
                for xval, yval in loader_val:
                    logalpha, mu, logsig = net(xval.to(self.device))
                    sig = torch.exp(logsig)

                    log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((yval.to(self.device)[:,None] - mu[:,:,None])/sig[:,:,None])**2
                    log_prob = torch.logsumexp(log_prob, 1)
                    loss = - log_prob.mean()
                    val_loss = loss.item()
                    val_losses.append(val_loss)

                if epoch % 1 == 0:
                    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, self.epochs, train_loss, val_loss))
                    
    def pred_photoz(self, plot=True):
        i=float(input('Enter flux i: '))
        g=float(input('Enter flux g: '))
        r=float(input('Enter flux r: '))
        z=float(input('Enter flux z: '))
        h=float(input('Enter flux h: '))
        j=float(input('Enter flux j: '))
        y=float(input('Enter flux y: '))
        df = pd.DataFrame(np.array([[i, g, r, z, h, j, y]]), columns=['i', 'g', 'r', 'z', 'h', 'j', 'y'])
        test_input=self._get_colorsdf(filetype='dataframe', df= df)
        
        logalpha, mu, logsig =  self.net_photoz(torch.Tensor(test_input.values).to(device))
        #Calculate alpha
        alpha = np.exp(logalpha.detach().cpu().numpy())
        #Calculate sigma
        sigma = np.exp(logsig.detach().cpu().numpy())
        #Calculate mu
        mu = mu.detach().cpu().numpy()
        #Calcuate zmean
        zmean = (alpha*mu).sum(1)
        #Create dataframe
        df = pd.DataFrame(np.c_[zmean], columns = ['z'])
         #Calculate and append error
        x = np.linspace(0, 1, 1000) #ya que filtramos catalogo a z<1
        galaxy_pdf = np.zeros(shape=x.shape)
        mean_pdf=0
        pick_galaxy=0
        for i in range(len(mu[pick_galaxy])):
            muGauss = mu[pick_galaxy][i]
            sigmaGauss = sigma[pick_galaxy][i]
            Gauss = stats.norm.pdf(x, muGauss, sigmaGauss)
            coefficients = alpha[pick_galaxy][i]
            mean_pdf= mean_pdf + muGauss*coefficients
            coefficients /= coefficients.sum()# in case these did not add up to 1
            Gauss= coefficients * Gauss
            galaxy_pdf = galaxy_pdf + Gauss


            #plt.plot(x, stats.norm.pdf(x, muGauss, sigmaGauss))
        galaxy_pdf_norm=galaxy_pdf/galaxy_pdf.sum()

        print("Mean of the distribution:", mean_pdf)  

        if plot== True:
            fig3 = plt.figure(figsize=(10, 6))
            plt.plot(x,galaxy_pdf_norm, color='black', label='galaxy_pdf_norm')
            # Add title and labels with LaTeX-style formatting
            plt.xlabel(f'$z$', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel(f' $p(z)$', fontsize=18)
            #z_true_line = plt.axvline(df.loc['zt'], color='r', linestyle=':', label='z_true')
            mean_pdf_line=plt.axvline(mean_pdf,color='g',linestyle='-', label='mean')
            plt.legend()
            plt.show()   

    #def get_training_distances(self, *args):
        # Function body...
  def train_clustering(self,*args):
  training_data=self.get_training_distances
  def train_mtl(self, *args):im
