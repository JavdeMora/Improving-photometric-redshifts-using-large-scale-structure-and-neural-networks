import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
import sys
import scipy.stats as stats
import json

sys.path.append('column_mapping.json')
sys.path.append('MTL_architecture.py')
from MTL_architecture import MTL_network
sys.path.append('plots_script.py')
from plots_script import plot_redshift_distribution

# Set the random seed for NumPy PyTorch and CUDA
np.random.seed(32)
torch.manual_seed(32)
torch.cuda.manual_seed(32)

class Photoz_MTL:
    """
    A class for training and predicting photometric redshifts using neural networks applying Multi-Task Learning (MTL).

    Args:
        pathfile_photometry (str): Path to the file containing photometric flux data.
        pathfile_distances (str): Path to the file containing real distances for a sky area.
        pathfile_drand (str): Path to the file containing random distances for a sky area.
        photoz_hlayers (int): Number of hidden layers in the photo-z prediction network.
        photoz_num_gauss (int): Number of output Gaussians in the photo-z prediction network.
        epochs (int): Number of epochs for training.
        lr (float): Learning rate for network training. Default is 1e-3.
        batch_size (int): Batch size for network training. Default is 100.
        min_sep (float): Minimum separation distance. Default is 0.03.
        max_sep (float): Maximum separation distance. Default is 26.
        nedges (int): Number of edges. Default is 8.

    Methods:
        __init__: Initializes the photo-z prediction model with provided parameters.
        _get_photometry_dataset: Reads photometry dataset from file.
        _get_colors: Calculates colors from photometry dataset.
        _get_loaders_photoz: Splits the data into train, validation, and test datasets and creates data loaders.
        training: Trains the photo-z prediction model.
        pred_photoz: Predicts redshift using flux inputs.
    """

    def __init__(self, 
                 pathfile_photometry,
                 pathfile_distances,
                 pathfile_drand,
                 photoz_hlayers, 
                 photoz_num_gauss,
                 epochs,
                 lr = 1e-3,
                 batch_size = 100,
                 min_sep = 0.03,
                 max_sep = 26,
                 nedges = 8
                ):
        
        
        self.net_MTL = MTL_network(photoz_hlayers, photoz_num_gauss).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nedges = nedges
        
        self.cat_photometry=self._get_photometry_dataset(pathfile_photometry)
        self.cat_colors =self._get_colors()
        self.d, self.drand = self._load_distances_array(pathfile_distances,pathfile_drand)
        self.MTL_df=self._get_MTL_df(self.cat_colors)
        
    def _get_photometry_dataset(self, pathfile, bands=['i', 'g', 'r', 'z', 'h', 'j', 'y']):
        """
        Reads a photometry dataset from a file.

        Args:
            pathfile (str): Path to the dataset file.
            bands (list of str): List of bands for photometry. Default is ['i', 'g', 'r', 'z', 'h', 'j', 'y'].

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
        script_dir = os.path.dirname('MTL_git.ipynb')  # Get directory of the current script  __file__ -> 'MTL_method.py'??
        json_file_path = os.path.join(script_dir, 'column_mapping.json')
        
        required_columns = set(bands)
        if ~required_columns.issubset(df.columns):
            # Rename columns based on the provided mapping
            with open(json_file_path, 'r') as json_file:
                column_mapping = json.load(json_file)
            df = df.rename(columns=column_mapping)
                        
            # Add error columns to corresponding magnitude columns
            for b in bands:
                if 'err_' + str(b) in df.columns:
                    df[b]=df[b]+df['err_'+b]    
            
            # Drop error columns and other unnecessary columns
            columns_to_drop = [col for col in df.columns if col.startswith('err_')]# + ['dec_gal', 'ra_gal']
            if 'dec_gal' in df.columns:
                columns_to_drop.append('dec_gal')
            if 'ra_gal' in df.columns:
                columns_to_drop.append('ra_gal')
            df = df.drop(columns=columns_to_drop, axis=1)
        
        #convert to magnitudes
        df[bands]= -2.5*np.log10(df[bands])-48.6
        if 'vis' in df.columns:
            df['vis']= -2.5*np.log10(df['vis'])-48.6
        #Drop NaN
        df=df.dropna(axis=0,how='any')
        
        #Filter data (Mag_i < 25, 0.5 < z < 0.6)
        df=df[df['i']< 25]
        if 'observed_redshift_gal' in df.columns:
            df=df[(df['observed_redshift_gal'] < 0.6) & (df['observed_redshift_gal'] > 0.5)]

        return df
    

    
    def _get_colors(self):
        """
        Calculates colors from photometry dataset.

        Returns:
            pandas.DataFrame: DataFrame containing calculated colors.
        """
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
    
    def _get_colors_pred(self, df):
        
        """
        Calculates colors from a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing photometric data.

        Returns:
            pandas.DataFrame: DataFrame containing calculated colors.
        """

        try:
            # Check if all required columns are present
            if all(col in df.columns for col in ['vis', 'observed_redshift_gal']):
                colors_df = pd.DataFrame({
                    'observed_redshift_gal': df['observed_redshift_gal'],
                    'Mag_i': df['vis'],
                    'g-r': df['g'] - df['r'],
                    'r-i': df['r'] - df['i'],
                    'i-z': df['i'] - df['z'],
                    'z-y': df['z'] - df['y'],
                    'y-j': df['y'] - df['j'],
                    'j-h': df['j'] - df['h'],
                    'distance': df['distance']
                })
            else:
                raise ValueError("Missing required columns")
        except KeyError:
            # Handle the case where some columns are missing
            colors_df = pd.DataFrame({
                'g-r': df['g'] - df['r'],
                'r-i': df['r'] - df['i'],
                'i-z': df['i'] - df['z'],
                'z-y': df['z'] - df['y'],
                'distance':df['distance']
            })   
        return colors_df
    
    def _load_distances_array(self, pathfile_distances, pathfile_drand):
        """
        Loads real and random distances from files.

        Args:
            pathfile_distances (str): Path to the distances file.
            pathfile_drand (str): Path to the random distances file.

        Returns:
            numpy array: Real distances for a sky area divided into 400 jackknifes.
            numpy array: Random distances for a sky area divided into 400 jackknifes.
        """
        # Load distances arrays
        d = np.load(pathfile_distances)
        drand = np.load(pathfile_drand)

        return d, drand
      
    def _get_MTL_df(self, colors_df):
        """
        Generates a DataFrame for Multi-Task Learning (MTL).

        Args:
            colors_df (pandas.DataFrame): DataFrame containing colors.

        Returns:
            pandas.DataFrame: DataFrame for MTL.
        """
        
        distA = self.d.flatten()
        distB = self.drand.flatten()
        
        d_sample=np.random.choice(distA,int(colors_df.shape[0]/2)+1)
        drand_sample=np.random.choice(distB,int(colors_df.shape[0]/2))
        # Create labels
        label_0 = np.zeros_like(d_sample)
        label_1 = np.ones_like(drand_sample)

        # Add labels as columns
        dsample_labeled = np.column_stack((d_sample, label_0))
        drandsample_labeled = np.column_stack((drand_sample, label_1))
        # Concatenate both arrays
        distances_array = np.concatenate((dsample_labeled, drandsample_labeled))
        #Add weights
        th = np.linspace(np.log10(self.min_sep), np.log10(self.max_sep), self.nedges)
        theta = 10**th * (1./60.)
        ratio_dists = np.array([len(distA[(distA>theta[k])&(distA<theta[k+1])]) / len(distB[(distB>theta[k])&(distB<theta[k+1])]) for k in range(len(theta)-1)])
        Ndd = np.array([len(distA[(distA>theta[k])&(distA<theta[k+1])]) for k in range(len(theta)-1)])
        Ndd = np.append(Ndd,len(distA[(distA>theta[-1])]))
        Pdd = Ndd / Ndd.sum()
        wdd = Pdd.max() / Pdd 
        ranges =  [(theta[k],theta[k+1]) for k in range(len(theta)-1)]
        ranges.append((theta[-1],1.1))
        ranges = np.array(ranges)
        weights = wdd
        range_idx = np.searchsorted(ranges[:, 1], distances_array[:,0], side='right')
        w = weights[range_idx]
        distances_array = np.c_[distances_array, w.reshape(len(w),1)]
        # Shuffle the rows
        np.random.shuffle(distances_array)
        # Create a new DataFrame from combined_array
        distances_df = pd.DataFrame(distances_array, columns=['distance', 'label','weight'])
        
        # Reset index of both DataFrames
        colors_df.reset_index(drop=True, inplace=True)
        distances_df.reset_index(drop=True, inplace=True)
        # Concatenate new_df with your existing DataFrame df
        MTL_df = pd.concat([colors_df, distances_df], axis=1)
        
        return MTL_df
    
    def _get_loaders(self, test_size, val_size, batch_size):
        """
        Splits the data into train, validation, and test datasets and creates data loaders.

        Args:
            test_size (float): Proportion of the dataset for the test split.
            val_size (float): Proportion of the training dataset for the validation split.
            batch_size (int): Batch size for the data loaders.

        Returns:
            DataLoader: Training data loader.
            DataLoader: Validation data loader.
            DataLoader: Test data loader.
        """
        # Split data into input (features) and target variables
        input_cols = ['g-r', 'r-i', 'i-z', 'z-y', 'y-j', 'j-h','distance']
        target_col = ['observed_redshift_gal','label']
        weight_col = 'weight'

        X = self.MTL_df[input_cols].values
        y = self.MTL_df[target_col].values  # Ensure target column has the correct shape
        z = self.MTL_df[weight_col].values.reshape(-1, 1)

        # Split data into train and test datasets
        X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=test_size)

        # Further split train dataset into train and validation datasets
        X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(X_train, y_train, z_train, test_size=val_size)

        # Convert data to PyTorch tensors
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train), torch.Tensor(z_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val), torch.Tensor(z_val))
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test), torch.Tensor(z_test))

        # Create data loaders
        loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return loader_train, loader_val, loader_test
    
    def training(self, test_size=0.2, val_size=0.25, *args):
        """
        Trains the photo-z prediction model.

        Args:
            test_size (float): Proportion of the dataset for the test split. Default is 0.2.
            val_size (float): Proportion of the training dataset for the validation split. Default is 0.25.
            *args: Additional arguments.

        Returns:
            None
        """
        # Get data loaders for training and validation sets
        self.loader_train, self.loader_val, self.loader_test = self._get_loaders(test_size, 
                                                                                 val_size, 
                                                                                 self.batch_size)

        # Transfer model to GPU
        self.net = self.net_MTL.cuda()

        # Initialize lists to store training losses and predicted parameters
        self.train_losses = [] 

        # Define optimizer and learning rate scheduler
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        CELoss = nn.CrossEntropyLoss(reduction='none')

        # Training loop
        for epoch in range(self.epochs):
            for datain, xeval, w in self.loader_train:
                optimizer.zero_grad() 
                logalpha, mu, logsig, dpred = self.net(datain.to(self.device))
                sig = torch.exp(logsig)
                
                #Compute clustering loss
                dloss = CELoss(dpred.squeeze(1),xeval[:,1].type(torch.LongTensor).cuda())
                wdloss = (w.cuda()*dloss).mean()
                
                # Compute photoz loss
                log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((xeval.to(self.device)[:,0].unsqueeze(1).unsqueeze(2) - mu[:,:,None])/sig[:,:,None])**2
                log_prob = torch.logsumexp(log_prob,1)
                phloss = - log_prob.mean()
                loss=wdloss+phloss
                
                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
                
                train_wdloss=wdloss.item()
                train_phloss=phloss.item()
                train_loss = loss.item()
                self.train_losses.append(train_loss)

            # Update learning rate
            scheduler.step()

            # Validation
            self.net.eval()
            self.val_losses = []

            with torch.no_grad():
                for xval, yval, w in self.loader_val:
                    logalpha, mu, logsig, dpred = self.net_MTL(xval.to(self.device))
                    sig = torch.exp(logsig)

                    # Compute log probabilities
                    log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((yval.to(self.device)[:,0].unsqueeze(1).unsqueeze(2) - mu[:,:,None])/sig[:,:,None])**2
                    log_prob = torch.logsumexp(log_prob,1)
                    phloss = - log_prob.mean()
                    loss = phloss
                    val_loss = loss.item()
                    self.val_losses.append(val_loss)

                # Print training and validation losses
                if epoch % 1 == 0:
                    print('Epoch [{}/{}], d Loss: {:.4f},ph Loss: {:.4f},Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, self.epochs, train_wdloss, train_phloss, train_loss, val_loss))
                    
     
    def pred_photoz(self, inputs_pathfile, all_rows=True, bands=['i', 'g', 'r', 'z', 'h', 'j', 'y'],plot=True):
        """
        Predicts redshift using flux inputs from a file.

        Args:
            inputs_pathfile (str): Path to the input file containing flux data.
            all_rows (bool): If True, consider all rows in the input data. If False, specify the range of rows to consider.
                Default is True.
            bands (list of str): List of bands to consider for flux data. Default is ['i', 'g', 'r', 'z', 'h', 'j', 'y'].
            plot (bool): Whether to plot the predicted redshift distribution. Default is True.

        Returns:
            pandas.DataFrame: DataFrame containing predicted redshifts.
        """
       
        inputs = self._get_photometry_dataset(inputs_pathfile, bands) #está bien el self.?
        inputs = self._get_colors_pred(inputs)
        if all_rows ==True:
            pass
        else:
            first_row = input('first row: ')
            last_row = input('last row: ')
            if last_row == first_row:
                inputs = inputs.iloc[int(first_row):int(first_row)+1]
            else:
                inputs = inputs.iloc[int(first_row):int(last_row)]
        # Drop unnecessary columns
        columns_to_drop =  set(['observed_redshift_gal','Mag_i'])
        if ~columns_to_drop.issubset(inputs.columns):
            inputs = inputs.drop(columns=columns_to_drop, axis=1)
        # Predict redshift
        logalpha, mu, logsig, dpred = self.net_MTL(torch.Tensor(inputs.to_numpy()).to(self.device))

        # Convert predictions to numpy arrays
        alpha = np.exp(logalpha.detach().cpu().numpy())
        sigma = np.exp(logsig.detach().cpu().numpy())
        mu = mu.detach().cpu().numpy()

        # Calculate mean redshift
        zmean = (alpha * mu).sum(1)
        
         # Create DataFrame for predicted redshifts
        df = pd.DataFrame(np.c_[zmean], columns=['z_mean'])
        
        if plot:
            plot_redshift_distribution(df, alpha, mu, sigma)
            
        return df
   
