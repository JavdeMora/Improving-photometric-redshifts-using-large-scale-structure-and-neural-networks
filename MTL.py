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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
import os

# %%
# Set the random seed for NumPy PyTorch and CUDA
np.random.seed(32)
torch.manual_seed(32)
torch.cuda.manual_seed(32)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
dataset = pd.read_csv('/nfs/pic.es/user/j/jdemora/MTL/MTL_df_weights.csv', sep =',', header=0, comment='#')

#Pasamos a colores
colors_df = pd.DataFrame(np.c_[dataset['observed_redshift_gal'],dataset['vis'],dataset['g']-dataset['r'],dataset['r']-dataset['i'],dataset['i']-dataset['z'], dataset['z']-dataset['y'], dataset['y']-dataset['j'],dataset['j']-dataset['h'], dataset['distance'],dataset['label'],dataset['weight']], columns=['observed_redshift_gal','Mag_i','g-r','r-i','i-z','z-y','y-j','j-h','distance','label','weight'])
colors_df.head()

# %%
test_size=0.25
train_dataset, test_dataset=train_test_split(colors_df, test_size=test_size)
# %%
val_size = 0.2
train_df, val_df = train_test_split(train_dataset, test_size=val_size)

# %%
from torch.utils.data import TensorDataset, DataLoader, Dataset

# %%
input_labs = ['g-r','r-i','i-z','z-y','y-j','j-h','distance']
target_lab = ['observed_redshift_gal','label']
weight_lab = ['weight']

# %%
train_input=torch.Tensor(train_df[input_labs].values)
val_input= torch.Tensor(val_df[input_labs].values)

train_target = torch.Tensor(train_df[target_lab].values)
val_target = torch.Tensor(val_df[target_lab].values)

train_dweight = torch.Tensor(train_df[weight_lab].values)
val_dweight = torch.Tensor(val_df[weight_lab].values)

# %%
i_test=test_dataset['Mag_i'].values
test_input=torch.Tensor(test_dataset[input_labs].values)
test_target=test_dataset[target_lab].values

# %%
train_dataset = TensorDataset(train_input,train_target,train_dweight)
val_dataset = TensorDataset(val_input,val_target, val_dweight)

# %%
batch_size=100
loader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size = batch_size, shuffle =False)

# %%
colors_df.head()


# %%
# %%
class network(torch.nn.Module):
    def __init__(self, nhidden, num_gauss):
        super().__init__()
        self.inputlay = torch.nn.Sequential(nn.Linear(7, 20),nn.LeakyReLU(0.1))
        
        params = np.linspace(20,200,nhidden)
        modules = []
        for k in range(nhidden-1):
            modules.append(nn.Linear(int(params[k]) ,int(params[k+1])))
            #Do I do the up and down?
            #Do I add a Dropout layer? solo en clustering
            #Do I add weights?
            #Or do I leave it as the photoz network is configurated for proper comparison?
            modules.append(nn.LeakyReLU(0.1))  
        self.hiddenlay = nn.Sequential(*modules)
        
        self.dpred = torch.nn.Sequential(nn.Linear(200 ,100), nn.LeakyReLU(0.1),nn.Dropout(0.01), nn.Linear(100 ,50), nn.LeakyReLU(0.1), nn.Linear(50 ,15), nn.LeakyReLU(0.1),nn.Linear(15, 2)) 
        self.logalphas = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100, num_gauss))                
        self.means = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100, num_gauss))
        self.logstds = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100 ,num_gauss))
        
    def forward(self, inp):
        
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        mu = self.means(x)
        logsig = self.logstds(x)
        logalpha=self.logalphas(x)
        dpred = self.dpred(x)
        
        logsig = torch.clamp(logsig,-5,5)
        logalpha = logalpha - torch.logsumexp(logalpha,1)[:,None] 


        
        return logalpha, mu, logsig, dpred 

# %%
torch.autograd.set_detect_anomaly(True)
CELoss = nn.CrossEntropyLoss(reduction='none')

def _network_training(net, epochs):
    train_losses = [] 
    alpha_list = []
    mu_list = []
    ztrue_list = []
    
    # Define optimizers for dloss and phloss separately
    learning_rates = [{'params': model.hiddenlay.parameters(), 'lr': 2e-3}, {'params': model.dpred.parameters(), 'lr': 1e-5}, 
                      {'params': model.logalphas.parameters(), 'lr': 2e-3},{'params': model.means.parameters(), 'lr': 2e-3},{'params': model.logstds.parameters(), 'lr': 2e-3}] 
    #optimizer = optim.SGD(learning_rates, momentum=0.9)
    optimizer_dloss =  optim.Adam(net.parameters(), lr = 1e-5 )
    #optimizer_phloss =  optim.Adam(net.parameters(), lr = 2e-3)
    
    #scheduler_dloss = torch.optim.lr_scheduler.StepLR(optimizer_dloss, step_size=1200, gamma=0.01)
    scheduler_phloss = torch.optim.lr_scheduler.StepLR(optimizer_phloss, step_size=50, gamma=0.1)
    
    for epoch in range(epochs):
        for datain, xeval in loader_train:
            optimizer_dloss.zero_grad() 
            optimizer_phloss.zero_grad() 
            
            logalpha, mu, logsig, dpred = net(datain.to(device))
            sig = torch.exp(logsig)
            
            # Loss function for dloss
            dloss = CELoss(dpred.squeeze(1), xeval[:, 1].type(torch.LongTensor).cuda()).mean()
            dloss.backward(retain_graph=True)
            optimizer_dloss.step()
            
            # Loss function for phloss
            log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((xeval.to(device)[:,0].unsqueeze(1).unsqueeze(2) - mu[:,:,None])/sig[:,:,None])**2
            log_prob = torch.logsumexp(log_prob, 1)
            phloss = -log_prob.mean()
            
            phloss.backward()
            optimizer_phloss.step()
            
            # Compute total loss for logging
            loss = dloss + phloss
            train_loss = loss.item()
            train_losses.append(train_loss)
            
        scheduler_dloss.step()
        scheduler_phloss.step()
        
        net.eval()
        val_losses = []
        with torch.no_grad():
            for xval, yval in loader_val:
                logalpha, mu, logsig, dpred = net(xval.to(device))
                sig = torch.exp(logsig)
                
                log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((yval[:,0].to(device).unsqueeze(1).unsqueeze(2)[:,None] - mu[:,:,None])/sig[:,:,None])**2
                log_prob = torch.logsumexp(log_prob, 1)
                phloss = -log_prob.mean()
                
                val_loss = phloss.item()
                val_losses.append(val_loss)
            
            if epoch % 1 == 0:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, epochs, train_loss, val_loss))


# %% [markdown]
# Trying to apply two learning rates

# %%
CELoss = nn.CrossEntropyLoss(reduction='none')
def _network_training(net,epochs):
    train_losses = [] 
    alpha_list = []
    mu_list = []
    ztrue_list = []
    optimizer = optim.Adam(net.parameters(), lr=2e-3) #chatgpt canviar args optim.Adam per tenir lr diferents per cluster i per photosz
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) #para ir cambiando el lr a medida q se itera
    for epoch in range(epochs):
        for datain, xeval, w in loader_train:
            optimizer.zero_grad() 
            logalpha, mu, logsig, dpred = net(datain.to(device))
            sig = torch.exp(logsig)
            #loss function: 
            dloss = CELoss(dpred.squeeze(1),xeval[:,1].type(torch.LongTensor).cuda())
            wdloss = (w.cuda()*dloss).mean()
            log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((xeval.to(device)[:,0].unsqueeze(1).unsqueeze(2) - mu[:,:,None])/sig[:,:,None])**2
            log_prob = torch.logsumexp(log_prob,1)
            phloss = - log_prob.mean()
            loss=wdloss+phloss
            loss.backward()
            optimizer.step()
       
            train_wdloss=wdloss.item()
            train_phloss=phloss.item()
            train_loss = loss.item()
            train_losses.append(train_loss)
        scheduler.step()
        
        
        net.eval()#desactivar algunas funciones de la red para poder evaluar resultados
        val_losses = []
        logalpha_list = []
        out_pred, out_true = [],[]
        with torch.no_grad():
            for xval, yval, w in loader_val:
                logalpha, mu, logsig, dpred= net(xval.to(device))
                sig = torch.exp(logsig)
               # dloss = CELoss(dpred.squeeze(1),yval[:,1].type(torch.LongTensor).cuda()).mean()
                log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((yval.to(device)[:,0].unsqueeze(1).unsqueeze(2) - mu[:,:,None])/sig[:,:,None])**2
                log_prob = torch.logsumexp(log_prob,1)
                phloss = - log_prob.mean()
                #loss=dloss+phloss
                loss=phloss

                val_loss = loss.item()
                val_losses.append(val_loss)
            
            if epoch % 1 == 0:
                print('Epoch [{}/{}], d Loss: {:.4f},ph Loss: {:.4f},Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, epochs,train_wdloss,train_phloss, train_loss, val_loss))

# %%
# %%
layers=3
num_gauss=5
net = network(layers,num_gauss).cuda()

# %%
#pretrained = torch.load('/nfs/pic.es/user/j/jdemora/MTL/MTL_results/model_MTL_25_4.pt')
#net.load_state_dict(pretrained)


# %%
# %%
#Ejecutar entrenamiento! Elegir numero de epochs
number_epochs=100
_network_training(net,number_epochs)

# %%
# %%
torch.save(net.state_dict(), '/nfs/pic.es/user/j/jdemora/MTL/MTL_results/model_MTL_25_4_weighted.pt')

# %%
#Extract parameters from network
logalpha, mu, logsig, dpred = net(test_input.to(device))

#Calculate alpha
alpha = np.exp(logalpha.detach().cpu().numpy())#.detach() es para quitar los gradients, .cpu() lo pasa a la cpu, .numpy() lo transforma en array
#alpha.sum(1) #comprobar que los pesos suman 1

#Calculate sigma
sigma = np.exp(logsig.detach().cpu().numpy())

#Calculate mu
mu = mu.detach().cpu().numpy()

#Calcuate zmean
zmean = (alpha*mu).sum(1)

s = nn.Softmax(1)
dp = s(dpred).detach().cpu().numpy()
#Create dataframe
df = pd.DataFrame(np.c_[zmean,test_target[:,0],i_test,dp[:,0],test_target[:,1]], columns = ['z','zt','i','dpred','d'])#crear dataframe columna con la z calculada a partir de la val_input y la ztrue de la val_target 
#Calculate and append error
df['zerr'] = (df.z -df.zt) / (1+df.zt) #creamos nueva columna con el error_z

df.sort_values(by='i').head()
df.to_csv('z_predicted_MTL_comparison.csv', index=False)

# %%
df.head()

# %%
import scipy.stats as stats
import os
def plot_photoz(df_test, nbins, xvariable, metric, type_bin='bin'):
    # Compute bin edges using quantiles
    bin_edges = stats.mstats.mquantiles(df_test[xvariable].values, np.linspace(0.1, 1, nbins))
    
    ydata, xlab = [], []

    for k in range(len(bin_edges) - 1):
        edge_min, edge_max = bin_edges[k], bin_edges[k + 1]
        mean_mag = (edge_max + edge_min) / 2

        if type_bin == 'bin':
            df_plot = df_test[(df_test[xvariable] > edge_min) & (df_test[xvariable] < edge_max)]
        elif type_bin == 'cum':
            df_plot = df_test[df_test[xvariable] < edge_max]
        else:
            raise ValueError("Only type_bin=='bin' for binned and 'cum' for cumulative are supported")

        xlab.append(mean_mag)
        if metric == 'sig68':#prec
            ydata.append(np.std(df_plot['zerr']))
        elif metric == 'mean':#bias
            ydata.append(np.mean(df_plot['zerr']))
        elif metric == 'nmad':
            ydata.append(nmad(df_plot['zerr']))
        elif metric == 'outliers':
            ydata.append(len(df_plot[np.abs(df_plot['zerr']) > 0.15]) / len(df_plot))

    # Plotting with improved style
    fig2 = plt.figure(figsize=(10, 6))  # Adjust the figure size
    plt.plot(xlab, ydata, ls='-', marker='o', color='navy', lw=2, label='')

    # Add title and labels with LaTeX-style formatting
    
    plt.title(f'{metric} $[\\Delta z]$ vs. {xvariable}', fontsize=20)
    plt.xlabel(f'${xvariable}$', fontsize=18)
    plt.ylabel(f'{metric} $[\\Delta z]$', fontsize=18)
    
    #for xy in zip(xlab, ydata):                                      
      #  plt.annotate('(%2f, %2f)' % xy, xy=xy, textcoords='data') 

    # Customize tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
    
    # Save the figure as before
    my_path = os.path.abspath('Bins3.ipynb')
    num_archivo = input('número archivo: ')
    if len(num_archivo) > 0:
        nombre2 = xvar + ' vs ' + met + ' ' +'MTL_weighted ' + str(num_archivo)
        my_file2 = nombre2 + '.pdf'
        fig2.savefig(os.path.join('/nfs/pic.es/user/j/jdemora/MTL/MTL_results/', my_file2))


# %%
xvar='i'
met='mean'
plot_photoz(df_test = df, nbins =10, xvariable=xvar,metric=met, type_bin='bin')

# %% jupyter={"outputs_hidden": true}
# %%
for j in range(0,20):
    pick_galaxy=j #5
    x = np.linspace(0, 1, 1000) #ya que filtramos catalogo a z<1
    galaxy_pdf = np.zeros(shape=x.shape)
    mean_pdf=0
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
    print("Galaxy number:", j)
    print("Mean of the distribution:", mean_pdf)
    print("Z true:", df.loc[pick_galaxy]['zt'])    


    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(x,galaxy_pdf_norm, color='black', label='galaxy_pdf_norm')
    # Add title and labels with LaTeX-style formatting
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(f'$z$', fontsize=18)
    plt.ylabel(f' $p(z)$', fontsize=18)
    z_true_line = plt.axvline(df.loc[pick_galaxy]['zt'], color='r', linestyle=':', label='z_true')
    mean_pdf_line=plt.axvline(mean_pdf,color='g',linestyle='-', label='mean')
    plt.legend()
    plt.show()

# %%
# %%
pick_galaxy=0 #114
x = np.linspace(0, 1, 1000) #ya que filtramos catalogo a z<1
galaxy_pdf = np.zeros(shape=x.shape)
mean_pdf=0
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
print("Z true:", df.loc[pick_galaxy]['zt'])    


fig3 = plt.figure(figsize=(10, 6))
plt.plot(x,galaxy_pdf_norm, color='black', label='galaxy_pdf_norm')
# Add title and labels with LaTeX-style formatting
plt.xlabel(f'$z$', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel(f' $p(z)$', fontsize=18)
z_true_line = plt.axvline(df.loc[pick_galaxy]['zt'], color='r', linestyle=':', label='z_true')
mean_pdf_line=plt.axvline(mean_pdf,color='g',linestyle='-', label='mean')
plt.legend()
plt.show()

# Save the figure as before
my_path = os.path.abspath('Bins3.ipynb')
num_archivo = input('número archivo: ')
if len(num_archivo) > 0:
    nombre = 'PDF_' + str(num_archivo)
    my_file3 = nombre + '.pdf'
    fig3.savefig(os.path.join('/nfs/pic.es/user/j/jdemora/Data_visualization/Plots/PDFs', my_file3))

# %%
