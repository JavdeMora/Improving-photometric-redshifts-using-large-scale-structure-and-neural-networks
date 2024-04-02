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
        plt.show()

#2PCF curves comparison
def plot_2PCF(pred_ratio, d, drand, min_sep, max_sep, nedges):
    # Calling real data
    distA = d.flatten()
    distB = drand.flatten()
    # Compute 2PCF using the real data
    th = np.linspace(np.log10(min_sep), np.log10(max_sep), nedges)
    thetac = 10**np.array([(th[i]+th[i+1])/2 for i in range(len(th)-1)])
    theta = 10**th * (1./60.)
    thetac = thetac/60
    ratio_dists = np.array([len(distA[(distA>theta[k])&(distA<theta[k+1])]) / len(distB[(distB>theta[k])&(distB<theta[k+1])]) for k in range(len(theta)-1)])

    network_color = 'crimson'
    true_color = 'navy'
    fig=plt.figure(figsize=(10, 6))
    # Plot predicted 2PCF 
    plt.errorbar(thetac_test, pred_ratio_jk.mean(0), pred_ratio_jk.std(0), color=network_color, ls='--', label='$p(DD)/p(RR)$ - Network', marker='.', markersize=8)

    # Plot 2PCF conventional method
    plt.plot(thetac, ratio_dists - 1, color=true_color, label='$p(DD)/p(RR)$ - True', linewidth=2, marker='o', markersize=8)

    plt.xscale('log')
    plt.ylabel(r'$w(\theta)$', fontsize=16)
    plt.xlabel(r'$\theta$', fontsize=16)
    plt.title('Comparison of Network Predictions and True Values', fontsize=18)
    plt.grid(which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.show()

