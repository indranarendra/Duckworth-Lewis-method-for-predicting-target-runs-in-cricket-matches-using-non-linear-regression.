import numpy as np
import pandas as pd
import scipy.optimize as sp
import matplotlib.pyplot as plt


def se_loss(par, args):
    total_se = 0
    l = par[10]
    inn = args[0]
    runs = args[1]
    overs = args[2]
    wickets = args[3]
    n = 0
    for i in range(len(inn)):
        if inn[i] == 1:
            n = n + 1
            z = runs[i]
            u = overs[i]
            w = wickets[i]
            z0 = par[w - 1]
            zp = z0 * (1 - np.exp(-1 * l * u / z0))
            total_se = total_se + (zp - z) ** 2
    mse = total_se/n
    return mse


def fit_parameters(inn, runs, overs, wickets):
    par = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 10]
    opt = sp.minimize(se_loss, par, args=[inn, runs, overs, wickets])
    return opt['fun'], opt['x']


if __name__ == "__main__":
    # reading the csv file
    dls_df = pd.read_csv('04_cricket_1999to2011.csv')

    # data pre processing
    inn = dls_df['Innings'].values
    runs = dls_df['Runs.Remaining'].values
    overs = dls_df['Total.Overs'].values - dls_df['Over'].values
    wickets = dls_df['Wickets.in.Hand'].values

    # optimization step
    mse, par = fit_parameters(inn, runs, overs, wickets)
    print(f'mse = {mse} \nparameters for avg score = {par[:10]} \nL = {par[10]}')

    # plotting
    u = np.arange(51)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, Item 1
    plt.title("Runs Scored")
    plt.xlim((0, 50))
    plt.ylim((0, 250))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 50, 100, 150, 200, 250])
    plt.xlabel('overs remaining')
    plt.ylabel('runs')
    plt.grid()
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#555b65', '#999e45', '#222a55']
    for i in range(len(par)-1):
        z = par[i] * (1 - np.exp(-par[10] * u / par[i]))
        plt.plot(u, z, c=colors[i], label='W'+str(i+1))
    plt.legend()

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, Item 2
    plt.title("Resources Remaining")
    plt.xlim((0, 50))
    plt.ylim((0, 100))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.grid()
    plt.xlabel('overs remaining')
    plt.ylabel('percentage of resources remaining')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#555b65', '#999e45', '#222a55']
    zn = par[9] * (1 - np.exp(-par[10] * 50 / par[9]))
    for i in range(len(par) - 1):
        z = par[i] * (1 - np.exp(-par[10] * u / par[i]))
        p = z/zn * 100
        plt.plot(u, p, c=colors[i], label='W'+str(i+1))
    plt.legend()

    plt.show()


