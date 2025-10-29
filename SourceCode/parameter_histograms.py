import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

original_parameters = ISONet.get_weights()

original_ws = original_parameters[0::2]
original_bs = original_parameters[1::2]

min_val, max_val = -1.0,+1.0

for original_w in original_ws:
    plt.figure(figsize=(8,5))
    kde = gaussian_kde(original_w.flatten())
    x = np.linspace(min_val,max_val,200)
    plt.hist(original_w.flatten(), bins=100, range = (min_val,max_val),color="r",alpha=1.0,density=True)
    plt.plot(x,kde(x),color="black")
    plt.xlim([min_val,max_val])
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Weights")
    plt.show()
##

ws = np.concatenate([w.flatten() for w in original_ws])
kde = gaussian_kde(ws.flatten())
x = np.linspace(min_val,max_val,200)
plt.hist(ws.flatten(), bins=100, range = (min_val,max_val),color="r",alpha=1.0,density=True)
plt.plot(x,kde(x),color="black")
plt.xlim([min_val,max_val])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Weights")
plt.show()

for original_b in original_bs:
    plt.figure(figsize=(8,5))
    kde = gaussian_kde(original_b.flatten())
    x = np.linspace(min_val,max_val,200)
    plt.hist(original_b.flatten(), bins=100, range = (min_val,max_val),color="b",alpha=1.0,density=True)
    plt.plot(x,kde(x),color="black")
    plt.xlim([min_val,max_val])
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Weights")
    plt.show()
##

ws = np.concatenate([w.flatten() for w in original_bs])
kde = gaussian_kde(ws.flatten())
x = np.linspace(min_val,max_val,200)
plt.hist(ws.flatten(), bins=100, range = (min_val,max_val),color="b",alpha=1.0,density=True)
plt.plot(x,kde(x),color="black")
plt.xlim([min_val,max_val])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Weights")
plt.show()