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


    def train_photoz(self, test_size=0.2, val_size=0.25, *args):
        """
        Train the photo-z prediction model.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the training dataset to include in the validation split.
            *args: Additional arguments.

        Returns:
            None
        """
        # Get data loaders for training and validation sets
        self.loader_train, self.loader_val, self.loader_test = self._get_loaders(test_size, 
                                                                                 val_size, 
                                                                                 self.batch_size)

        # Transfer model to GPU
        self.net = self.net_photoz.cuda()

        # Initialize lists to store training losses and predicted parameters
        self.train_losses = [] 

        # Define optimizer and learning rate scheduler
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        # Training loop
        for epoch in range(self.epochs):
            for datain, xeval in self.loader_train:
                optimizer.zero_grad() 
                logalpha, mu, logsig = self.net(datain.to(self.device))
                sig = torch.exp(logsig)

                # Compute log probabilities
                log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((xeval.to(self.device)[:,None] - mu[:,:,None])/sig[:,:,None])**2
                log_prob = torch.logsumexp(log_prob, 1)
                loss = - log_prob.mean()

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                self.train_losses.append(train_loss)

            # Update learning rate
            scheduler.step()

            # Validation
            net.eval()
            self.val_losses = []

            with torch.no_grad():
                for xval, yval in loader_val:
                    logalpha, mu, logsig = net(xval.to(self.device))
                    sig = torch.exp(logsig)

                    # Compute log probabilities
                    log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((yval.to(self.device)[:,None] - mu[:,:,None])/sig[:,:,None])**2
                    log_prob = torch.logsumexp(log_prob, 1)
                    loss = - log_prob.mean()
                    val_loss = loss.item()
                    self.val_losses.append(val_loss)

                # Print training and validation losses
                if epoch % 1 == 0:
                    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, self.epochs, train_loss, val_loss))
                    
                    
                    
    def pred_photoz(self, test_colors,plot=True):
        """
        Predict redshift using flux inputs.

        Args:
            plot (bool): Whether to plot the predicted redshift distribution. Default is True.

        Returns:
            None
        """
        # Predict redshift
        logalpha, mu, logsig = self.net_photoz(torch.Tensor(test_colors).to(self.device))

        # Convert predictions to numpy arrays
        alpha = np.exp(logalpha.detach().cpu().numpy())
        sigma = np.exp(logsig.detach().cpu().numpy())
        mu = mu.detach().cpu().numpy()

        # Calculate mean redshift
        zmean = (alpha * mu).sum(1)

        # Create DataFrame for predicted redshifts
        df = pd.DataFrame(np.c_[zmean], columns=['z'])

        # Calculate and append error
        x = np.linspace(0, 1, 1000)
        galaxy_pdf = np.zeros(shape=x.shape)
        alpha= alpha / alpha.sum()

        for i in range(len(mu[0])):
            muGauss = mu[0][i]
            sigmaGauss = sigma[0][i]
            Gauss = stats.norm.pdf(x, muGauss, sigmaGauss)
            coeff = alpha[0][i]
            Gauss = coeff * Gauss
            galaxy_pdf += Gauss

        galaxy_pdf_norm = galaxy_pdf / galaxy_pdf.sum()

        # Print mean of the distribution
        print("Mean of the distribution:", mean_pdf)

        # Plot predicted redshift distribution
        if plot:
            fig3 = plt.figure(figsize=(10, 6))
            plt.plot(x, galaxy_pdf_norm, color='black', label='galaxy_pdf_norm')
            plt.xlabel(f'$z$', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel(f'$p(z)$', fontsize=18)
            mean_pdf_line = plt.axvline(mean_pdf, color='g', linestyle='-', label='mean')
            plt.legend()
            plt.show() 
            
        return zmean, galaxy_pdf_norm


    def pred_photoz_arr(self, test_colors,plot=True):
        """
        Predict redshift using flux inputs.

        Args:
            plot (bool): Whether to plot the predicted redshift distribution. Default is True.

        Returns:
            None
        """
        # Predict redshift
        logalpha, mu, logsig = self.net_photoz(torch.Tensor(test_colors).to(self.device))

        # Convert predictions to numpy arrays
        alpha = np.exp(logalpha.detach().cpu().numpy())
        sigma = np.exp(logsig.detach().cpu().numpy())
        mu = mu.detach().cpu().numpy()

        # Calculate mean redshift
        zmean = (alpha * mu).sum(1)

        return zmean
