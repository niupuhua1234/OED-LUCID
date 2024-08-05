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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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

def custom_guide(dosage,time, senescent=None):
    B_loc = pyro.param('B_loc', torch.tensor(10.),
                         constraint=constraints.positive)
    B_scale = pyro.param('B_scale', torch.tensor(10.),
                         constraint=constraints.positive)
    # weights_loc = pyro.param('weights_loc', torch.zeros(3))
    # weights_scale = pyro.param('weights_scale',torch.ones(3),
    #                            constraint=constraints.positive)
    # B = pyro.sample("B", dist.Beta(B_loc, B_scale))
    # b_0 = pyro.sample("b0", dist.Normal(weights_loc[0], weights_scale[0]))
    # b_1 = pyro.sample("b1", dist.Normal(weights_loc[1], weights_scale[1]))
    # b_2 = pyro.sample("b2", dist.Normal(weights_loc[2], weights_scale[2]))

    weights_loc = pyro.param('weights_loc',lambda: 10*torch.ones(3),
                               constraint=constraints.positive)
    weights_scale = pyro.param('weights_scale',lambda: 10*torch.ones(3),
                               constraint=constraints.positive)
    B = pyro.sample("B", dist.Beta(B_loc, B_scale))
    b_0 = pyro.sample("b0", dist.Beta(weights_loc[0], weights_scale[0]))
    b_1 = pyro.sample("b1", dist.Beta(weights_loc[1], weights_scale[1]))
    b_2 = pyro.sample("b2", dist.Beta(weights_loc[2], weights_scale[2]))
    return {B:"B","b_0": b_0, "b_1": b_1, "b_2": b_2}


#pyro.render_model(custom_guide, model_args=(is_cont_africa, ruggedness, log_gdp), render_params=True)
#d_trains=[0.1680, 0.5680, 0.9680, 1.3680, 1.7680, 2.1680, 2.5680, 2.9680]
d_trains=[0.1680,1.1680,3.1680,6.168,12.168,24.168,48.168, 96.168,192.168,384.168,768.168,1536.168,3072.168]
pred_sens_2=[]
qols_2=[]
pred_sens_5=[]
qols_5=[]
pred_sens_10=[]
qols_10=[]
sims=[]
datas=[]
for d_train in d_trains:
    print("Dosage level: {}".format(d_train))
    # load data
    B, beta0, beta1, beta2 = 0.5, 0.05, 0.003, 0.01  # 0.5,0.1,0.03,0.08# ground-truth
    t = np.arange(50)
    r_d = lambda d: (beta0 + beta1 * d + beta2 * (d ** 2))
    senescent = 100 / (1 + 100 * B * np.exp(-r_d(d_train) * t))
    data = senescent + 1.* np.random.randn(50)

    pyro.clear_param_store()
    adam = ClippedAdam({"lr": 0.02})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, custom_guide, adam, elbo)

    losses = []
    time   =      torch.arange(50,dtype=torch.float )
    data   =      torch.tensor(data, dtype=torch.float )
    dosage =      d_train*torch.ones(50,dtype=torch.float )
    for step in range(2000):  # Consider running for more steps.
        loss = svi.step(dosage,time,data)
        losses.append(loss)
        if step % 100 == 0:
            print("Elbo loss: {}".format(loss))

    # plt.figure(figsize=(5, 2))
    # plt.plot(losses)
    # plt.xlabel("SVI step")
    # plt.ylabel("ELBO loss")


    # for name, value in pyro.get_param_store().items():
    #     print(name, pyro.param(name).data.cpu().numpy())
    #
    # # grab the learned variational parameters
    # with pyro.plate("samples", 800, dim=-1):
    #     samples = custom_guide(dosage,time)
    # fig = plt.figure(figsize=(10, 6))
    # sns.histplot(samples['B'].detach().cpu().numpy(), kde=True, stat="density", label="African nations")
    # plt.show()

    predictive = pyro.infer.Predictive(model, guide=custom_guide, num_samples=800)
    d_test=10*torch.ones(1,dtype=torch.float )
    t_test=2*torch.ones(1,dtype=torch.float )
    svi_samples = predictive(d_test,t_test)
    svi_sens = svi_samples["obs"]
    mean_qol=(-(svi_sens - svi_sens.mean(0)) ** 2).mean()

    pred_sens_2.append(svi_sens)
    qols_2.append(mean_qol)

    d_test=10*torch.ones(1,dtype=torch.float )
    t_test=5*torch.ones(1,dtype=torch.float )
    svi_samples = predictive(d_test,t_test)
    svi_sens = svi_samples["obs"]
    mean_qol=(-(svi_sens - svi_sens.mean(0)) ** 2).mean()

    pred_sens_5.append(svi_sens)
    qols_5.append(mean_qol)

    d_test = 10 * torch.ones(1, dtype=torch.float)
    t_test = 10 * torch.ones(1, dtype=torch.float)
    svi_samples = predictive(d_test, t_test)
    svi_sens = svi_samples["obs"]
    mean_qol = (-(svi_sens - svi_sens.mean(0)) ** 2).mean()

    pred_sens_10.append(svi_sens)
    qols_10.append(mean_qol)

    sims.append(predictive(d_train,time)["obs"].mean(0))
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
for i,c in zip(sims,colors):
    plt.plot(i,color=c)

plt.legend(d_trains)
