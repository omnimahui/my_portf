import matplotlib.pyplot as plt
import numpy as np


def plot_min_var_frontier(mu, Cov): 
    A,B,C = compute_ABC(mu, Cov)
    y = np.linspace(-1,5,100)
    x = (A*y*y-2*B*y+C)/(A*C-B*B)
    x = np.sqrt(x)
    plt.plot(x,y, color='black', lw=2.0)
def compute_ABC(mu, Cov):
    Cov_inv = np.linalg.inv(Cov)
    ones = np.ones(Cov.shape[0])
    A = ones @ Cov_inv @ ones
    B = ones @ Cov_inv @ mu
    C = mu @ Cov_inv @ mu
    return A,B,C

def plot_Capital_Allocation_Line(rf, mu, Cov): 
    A,B,C = compute_ABC(mu, Cov)
    x = np.linspace(0,1,100)
    y = rf + x*(C-2*B*rf+A*rf**2)**0.5
    plt.plot(x,y, color='black', lw=2.5)

def plot_points(mu, sigma, stocks, date):
    plt.figure(figsize=(8,6))
    plt.title(date)
    plt.scatter(sigma, mu, c='black') 
    plt.xlim(0,2)
    plt.ylim(-1,5)
    plt.ylabel('mean')
    plt.xlabel('std dev')
    for i, stock in enumerate(stocks):
        plt.annotate(stock, (sigma[i], mu[i]), ha='center', va='bottom', weight='bold') 

