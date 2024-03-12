        #Plot 2PCF 
        if plot:
            #Calling real data
            distA = self.d.flatten()
            distB = self.drand.flatten()
            #Compute 2PCF using the real data
            th = np.linspace(np.log10(self.min_sep), np.log10(self.max_sep), self.nedges)
            thetac = 10**np.array([(th[i]+th[i+1])/2 for i in range(len(th)-1)])
            theta = 10**th * (1./60.)
            thetac = thetac/60
            ratio_dists = np.array([len(distA[(distA>theta[k])&(distA<theta[k+1])]) / len(distB[(distB>theta[k])&(distB<theta[k+1])]) for k in range(len(theta)-1)])
            
            network_color = 'crimson'
            true_color = 'navy'
            fig=plt.figure(figsize=(10, 6))
            #Plot predicted 2PCF 
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