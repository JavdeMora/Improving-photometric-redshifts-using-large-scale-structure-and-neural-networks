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
import scipy.stats as stats

sys.path.append('Photo_z_architecture.py')
from Photo_z_architecture import photoz_network



class MTL_photoz:
    def __init__(self, 
                 photoz_hlayers, 
                 photoz_num_gauss,
                 epochs,
                 lr=1e-3,
                 batch_size=100, 
                 pathfile='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/catalogues/FS2.csv'
                ):
        
        
        self.net_photoz = photoz_network(photoz_hlayers, photoz_num_gauss).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        
        cat_photometry=self._get_photometry_dataset(pathfile)
        self.cat_photometry=cat_photometry
        self.cat_colors =self._get_colors()
        
        
        
        cat=self._get_colors(pathfile=pathfile)
        self.cat=cat
        self.test_input=torch.Tensor(self.cat.loc[0][['g-r','r-i','i-z','z-y','y-j','j-h']].values)# esto es solo para testear
        
        
    def _get_photometry_dataset(self, pathfile, bands=['i', 'g', 'r', 'z', 'h', 'j', 'y']):
        """
        Read photometry dataset from file.

        Args:
            pathfile (str): Path to the dataset file.

        Returns:
            pandas.DataFrame: Photometry dataset.
        """
        # Determine the file extension
        file_extension = os.path.splitext(pathfile)[1]

        # Read the dataset based on the file type
        if file_extension == '.csv':
            df = pd.read_csv(str(pathfile), sep=',', header=0, comment='#')
        elif file_extension == '.parquet':
            df = pd.read_parquet(str(pathfile), sep=',', header=0, comment='#')
        else:
            raise ValueError("Only filetypes '.csv' and '.parquet' are supported")
            
        # Check if all required columns are present in the DataFrame
        column_mapping = 
        required_columns = set(bands)
        if ~required_columns.issubset(df.columns):
            # Rename columns based on the provided mapping
            with open(json_file_path, 'r') as json_file:
                column_mapping = json.load('column_mapping.json')
            df = df.rename(columns=column_mapping)
                        
            # Add error columns to corresponding magnitude columns
            for b in bands:
                df[b]=df[b]+df['err_'+b]    
            
            # Drop error columns and other unnecessary columns
            columns_to_drop = [col for col in df.columns if df.startswith('err_')] + ['dec_gal', 'ra_gal']
            df = df.drop(columns=columns_to_drop, axis=1)    
            
        #convert to magnitudes
        df[bands]= -2.5*np.log10(df[bands])-48.6
        if 'vis' in df.columns:
            df['vis']= -2.5*np.log10(df['vis'])-48.6
        #Drop NaN
        df=df.dropna(axis=0,how='any')
        
        #Filter data (Mag_i < 25, z<1)
        df=df[df['i']< 25]
        if 'observed_redshift_gal' in df.columns:
            df=df[df['observed_redshift_gal']<1]

        return df
    
    def _get_colors(self):

        try:
            # Check if all required columns are present
            if all(col in self.cat_photometry.columns for col in ['vis', 'observed_redshift_gal']):
                colors_df = pd.DataFrame({
                    'observed_redshift_gal': self.cat_photometry['observed_redshift_gal'],
                    'Mag_i': self.cat_photometry['vis'],
                    'g-r': self.cat_photometry['g'] - self.cat_photometry['r'],
                    'r-i': self.cat_photometry['r'] - self.cat_photometry['i'],
                    'i-z': self.cat_photometry['i'] - self.cat_photometry['z'],
                    'z-y': self.cat_photometry['z'] - self.cat_photometry['y'],
                    'y-j': self.cat_photometry['y'] - self.cat_photometry['j'],
                    'j-h': self.cat_photometry['j'] - self.cat_photometry['h']
                })
            else:
                raise ValueError("Missing required columns")
        except KeyError:
            # Handle the case where some columns are missing
            colors_df = pd.DataFrame({
                'g-r': self.cat_photometry['g'] - self.cat_photometry['r'],
                'r-i': self.cat_photometry['r'] - self.cat_photometry['i'],
                'i-z': self.cat_photometry['i'] - self.cat_photometry['z'],
                'z-y': self.cat_photometry['z'] - self.cat_photometry['y']
            })   
        return colors_df
        
    def _get_loaders_photoz(self, test_size, val_size, batch_size):
        """
        Split the data into train, validation, and test datasets and create data loaders.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the training dataset to include in the validation split.
            batch_size (int): The batch size for the data loaders.

        Returns:
            DataLoader: Training data loader.
            DataLoader: Validation data loader.
            DataLoader: Test data loader.
        """
        # Split data into input (features) and target variables
        input_cols = ['g-r', 'r-i', 'i-z', 'z-y', 'y-j', 'j-h']
        target_col = 'observed_redshift_gal'

        X = self.cat_colors[input_cols].values
        y = self.cat_colors[target_col].values.reshape(-1, 1)  # Ensure target column has the correct shape

        # Split data into train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Further split train dataset into train and validation datasets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

        # Convert data to PyTorch tensors
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

        # Create data loaders
        loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return loader_train, loader_val, loader_test


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
        test_input=self._get_colors(filetype='dataframe', df= df)
        
        logalpha, mu, logsig =  self.net_photoz(torch.Tensor(test_input.values).to(self.device))
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
    #def get_training_distances(self, *args):
        # Function body...
  def train_clustering(self,*args):
  training_data=self.get_training_distances
  def train_mtl(self, *args):im
