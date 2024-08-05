import  pyro
import torch.distributions.constraints as constraints
import torch
import matplotlib.pyplot as plt
import pyro.distributions as dist
import pandas as pd
import numpy as np
import os
from pyro.optim import ClippedAdam
import seaborn as sns
from pyro.infer import MCMC, NUTS
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#torch.set_rng_state()
#model
def model(dosage,time, senescent=None):
    B = pyro.sample("B", dist.Beta(10., 10.))
    # b_0 = pyro.sample("b0", dist.Normal(0., 1.))
    # b_1 = pyro.sample("b1", dist.Normal(0., 1.))
    # b_2 = pyro.sample("b2", dist.Normal(0., 1.))

    b_0 = pyro.sample("b0",  dist.Beta(10., 10.))
    b_1 = pyro.sample("b1",  dist.Beta(10., 10.))
    b_2 = pyro.sample("b2",  dist.Beta(10., 10.))
    mean= 100/(1+100*B*torch.exp(-(b_0/10+b_1/100*dosage+b_2/10*(dosage**2))*time))
    with pyro.plate("data", len(time)):
        return pyro.sample("obs", dist.Normal(mean, 1.), obs=senescent)

#pyro.render_model(model, model_args=(is_cont_africa, ruggedness, log_gdp), render_distributions=True)

# Utility function to print latent sites' quantile information.
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats


#pyro.render_model(custom_guide, model_args=(is_cont_africa, ruggedness, log_gdp), render_params=True)
d_trains=[5.1680,6.168,12.168,24.168,48.168, 96.168,192.168,384.168,768.168,1536.168,3072.168]
pred_sens_2=[]
qols_2=[]
pred_sens_5=[]
qols_5=[]
pred_sens_10=[]
qols_10=[]
hmcs=[]
sims=[]
datas=[]
from pyro.infer.autoguide import init_to_median
def main(data):
    pyro.clear_param_store()
    nuts_kernel = NUTS(model,adapt_step_size=True,init_strategy=init_to_median)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)

    time   =      torch.arange(50,dtype=torch.float )
    data   =      torch.tensor(data, dtype=torch.float )
    dosage =      d_train*torch.ones(50,dtype=torch.float )

    mcmc.run(dosage,time,data)
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    for site, values in summary(hmc_samples).items():
        print("Site: {}".format(site))
        print(values, "\n")

    predictive = pyro.infer.Predictive(model,posterior_samples=hmc_samples)
    d_test=10*torch.ones(1,dtype=torch.float )
    t_test=2*torch.ones(1,dtype=torch.float )
    svi_samples = predictive(d_test,t_test)
    svi_sens_2 = svi_samples["obs"]
    mean_qol_2=(-(svi_sens_2 - svi_sens_2.mean(0)) ** 2).mean()

    d_test=10*torch.ones(1,dtype=torch.float )
    t_test=5*torch.ones(1,dtype=torch.float )
    svi_samples = predictive(d_test,t_test)
    svi_sens_5 = svi_samples["obs"]
    mean_qol_5=(-(svi_sens_5 - svi_sens_5.mean(0)) ** 2).mean()


    sim=predictive(d_train,time)["obs"].mean(0)

    t_indexs=[]
    for B,beta0,beta1,beta2 in zip(hmc_samples['B'],hmc_samples['b0'],hmc_samples['b1'],hmc_samples['b2']):
        r_d = lambda d: (beta0 / 10 + beta1 / 100 * d + beta2 / 10 * (d ** 2))
        senescent=100 / (1 + 100 * B * np.exp(-r_d(10) * t))
        t_indexs.append(np.argmin(np.abs(senescent - np.max(senescent) / 2)))


    svi_sens_10 = torch.tensor(t_indexs)
    mean_qol_10=(-(svi_sens_10 - svi_sens_10.mean(0)) ** 2).mean()

    return svi_sens_2,mean_qol_2,svi_sens_5,mean_qol_5,svi_sens_10,mean_qol_10,sim,hmc_samples

for d_train in d_trains:
    print("Dosage level: {}".format(d_train))
    # load data
    B, beta0, beta1, beta2 = 0.5, 0.05, 0.003, 0.01  # 0.5,0.1,0.03,0.08# ground-truth
    t = np.arange(50)
    r_d = lambda d: (beta0 + beta1 * d + beta2 * (d ** 2))
    senescent = 100 / (1 + 100 * B * np.exp(-r_d(d_train) * t))
    data = senescent + 0.* np.random.randn(50)
    svi_sens_2,mean_qol_2,svi_sens_5,mean_qol_5,svi_sens_10,mean_qol_10,sim,hmc_samples=main(data)
    pred_sens_2.append(svi_sens_2)
    qols_2.append(mean_qol_2)
    pred_sens_5.append(svi_sens_5)
    qols_5.append(mean_qol_5)
    pred_sens_10.append(svi_sens_10)
    qols_10.append(mean_qol_10)
    hmcs.append(hmc_samples)
    sims.append(sim)
    datas.append(data)


pred_mean=torch.stack(pred_sens_2,dim=-1).squeeze().mean(0)
pred_std=torch.stack(pred_sens_2,dim=-1).squeeze().std(0)
plt.plot(torch.tensor(d_trains),pred_mean,color='blue')
plt.errorbar(torch.tensor(d_trains),pred_mean,pred_std,color='blue',alpha=0.2)
#plt.fill_between(torch.tensor(d_trains),pred_mean-pred_std,pred_mean+pred_std,alpha=0.2)
plt.xlabel("dosage levels/mGy ")
plt.ylabel("senescent/%")
plt.xscale('log')


pred_mean=torch.stack(qols_2)
plt.plot(torch.tensor(d_trains),pred_mean,color='blue')
#plt.fill_between(torch.tensor(d_trains),pred_mean-pred_std,pred_mean+pred_std,alpha=0.2)
plt.xlabel("dosage levels/mGy ")
plt.ylabel("Qol")
plt.xscale('log')


colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i,c,d_train in zip(hmcs,colors,d_trains):
    B, beta0, beta1, beta2=i['B'].mean(),i['b0'].mean(),i['b1'].mean(),i['b2'].mean()
    t = np.arange(50)
    r_d = lambda d: (beta0/10 + beta1/100 * d + beta2/10 * (d ** 2))
    senescent = 100 / (1 + 100 * B * np.exp(-r_d(d_train) * t))
    plt.plot(senescent,color=c)

for c,d_train in zip(colors,d_trains):
    B, beta0, beta1, beta2=0.5, 0.05, 0.003, 0.01
    t = np.arange(50)
    r_d = lambda d: (beta0 + beta1 * d + beta2 * (d ** 2))
    senescent = 100 / (1 + 100 * B * np.exp(-r_d(d_train) * t))
    plt.plot(senescent,color=c,linestyle='--')
