import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Photo-z probability distribution plot
def plot_redshift_distribution(df, alpha, mu, sigma):
    x = np.linspace(0, 1, 1000)
    for j in range(len(df)):
        pick_galaxy = j
        galaxy_pdf = np.zeros(shape=x.shape)
        mean_pdf = 0
        for i in range(len(mu[0])):
            muGauss = mu[pick_galaxy][i]
            sigmaGauss = sigma[pick_galaxy][i]
            Gauss = stats.norm.pdf(x, muGauss, sigmaGauss)
            coeff = alpha[pick_galaxy][i]
            Gauss = coeff * Gauss
            galaxy_pdf += Gauss
            mean_pdf = mean_pdf + muGauss * coeff
        galaxy_pdf_norm = galaxy_pdf / galaxy_pdf.sum()
        fig3 = plt.figure(figsize=(10, 6))
        plt.plot(x, galaxy_pdf_norm, color='black', label='galaxy_pdf_norm')
        plt.xlabel('$z$', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylabel('$p(z)$', fontsize=18)
        mean_pdf_line = plt.axvline(df['z_mean'].values[j], color='g', linestyle='-', label='mean')
        plt.legend()
        print('Z_mean = {:.4f} for galaxy number {}'.format(df['z_mean'].values[j],pick_galaxy))
        plt.show()

#2PCF curves comparison
def plot_2PCF(pred_ratio, d, drand, min_sep, max_sep, nedges):
    def estimate(x):
        mean_x = np.mean(x,0)

        means = []
        for ii in range(len(x)):
            means.append(np.mean(np.delete(x,ii,axis=0),0))

        means = np.array(means)

        err = np.sqrt((len(x)-1)*np.sum((means - mean_x)**2,0))
        return mean_x, err
    # Calling real data
    distA = d.flatten()
    distB = drand.flatten()
    # Compute 2PCF using the real data
    th = np.linspace(np.log10(min_sep), np.log10(max_sep), nedges)
    thetac = 10**np.array([(th[i]+th[i+1])/2 for i in range(len(th)-1)])
    theta = 10**th * (1./60.)
    thetac = thetac/60
    ratio_dists = np.array([len(distA[(distA>theta[k])&(distA<theta[k+1])]) / len(distB[(distB>theta[k])&(distB<theta[k+1])]) for k in range(len(theta)-1)])
    
    mean_nn, std_nn = estimate(pred_ratio)
    plt.figure(figsize=(10, 10))
    plt.plot(thetac, ratio_dists - 1, color='navy',markersize=6, label='Analytical estimation', marker='o', zorder=3)
    plt.plot(thetac, mean_nn, marker='o', markersize=6, color='#E63946', label='NN estimation', zorder=3)

    # Jackknife mean
    plt.plot(thetac, pred_ratio.mean(0), ls='-', lw=1.5, color='#E63946', zorder=2)
    # Error bars
    plt.fill_between(thetac, mean_nn - std_nn, mean_nn + std_nn, color='#BFF6C3', alpha=0.5, label='Error', zorder=1)

    plt.xscale('log')
    plt.ylabel(r'$w(\theta)$', fontsize=16)
    plt.xlabel(r'$\theta$', fontsize=16)
    plt.grid(which='both', linestyle='--', linewidth=0.5, color='#A8DADC')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)

    plt.title('Two Point Correlation Function (2PCF) estimation', fontsize=18, pad=20)

    plt.tight_layout()
    plt.show()
