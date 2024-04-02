import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
