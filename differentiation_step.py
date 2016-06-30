import numpy as np
from scipy.optimize import minimize
from random import getrandbits
import matplotlib.pyplot as plt


# Helper functions.
def softmax(x):
    e_x = np.exp(x)
    out = e_x / e_x.sum()
    return out

def vect_length(m):
    return np.linalg.norm(m, axis=1).reshape(-1, 1)

def square_vect_length(m):
    return np.power(m, 2).sum()

def dist(x, y):
    return vect_length(x-y)

def square_dist(x, y):
    return square_vect_length(x-y)

# calculation of the sigmas: xis, yis, and dists are assumed to be aligned 
# numpy arrays of indices and distances. 
def calculate_all_sigmas_and_pjis(xis, yis, dists, perplexity = 50, tolerance = 0.001):
    max_xi = np.max(xis)
    sigmas = np.zeros(max_xi+1)
    pjis = np.zeros_like(xis, dtype='float')
    for xi in range(max_xi+1):
        matches = (xis == xi)
        sigmas[xi], pjis[matches] = calculate_sigma_and_pjis_for_fixed_xi(np.power(dists[matches], 2), perplexity, tolerance)
    return sigmas, pjis
    
def pjis_from(sigma, square_dists):
    exps = np.exp(-square_dists / sigma)
    return exps / np.sum(exps)

def calculate_sigma_and_pjis_for_fixed_xi(square_dists, perplexity = 50, tolerance = 0.001):
    def to_minimize(s):  # actually learns 2*sigma^2
        p_jis = pjis_from(s, square_dists)
        return np.power(perplexity - (np.sum(np.log2(p_jis)) / len(square_dists)), 2)
    sigma=minimize(to_minimize, 5, tol=tolerance, options={'maxiter':1000}).x
    result = np.sqrt(sigma) / 2
    pji_values = pjis_from(sigma, square_dists)
    return result, pji_values


# Derivative stuff

num_pts = 500
num_batches = 10000
batch_size = 10
init_learning_rate = 1000.0
gamma = 7
num_corruptions = 10
output_dim = 2
num_nns = 2


# input (probably not most efficient way of doing this, but it's not meant to last):
### input_embeddings = np.array([[np.sin(x*np.pi*2/num_pts), np.cos(x*np.pi*2/num_pts)] for x in range(num_pts)])
def shifted(n):
    return range(n, num_pts)+range(n)

### CIRCLE:
#link_data = [np.array(range(num_pts)*num_nns), np.array([x for i in range(num_nns) for x in shifted(i+1)])]
#link_distances = np.concatenate([(1+i)*np.ones(num_pts) for i in range(num_nns)])

### LINE:
link_data = [np.array(range(num_pts-1)), np.array(range(1, num_pts))]
link_distances = np.ones(num_pts-1)

no_skip = 0

# computed once from input:
sigmas, pjis = calculate_all_sigmas_and_pjis(link_data[0], link_data[1], link_distances)
sigma2s, pijs = calculate_all_sigmas_and_pjis(link_data[1], link_data[0], link_distances)
# print sigmas
probabilities = softmax(pjis*pijs / (2.0*num_pts))
hist = np.histogram(np.power(np.concatenate([link_data[0], link_data[1]]), 0.75), num_pts)[0]
negative_sample_probabilities = softmax(hist)

embeddings = np.random.randn(num_pts, output_dim)
num_links = link_data[0].size

def positive_gradient(x, y):
    diff = x-y
    diff_square_norm = square_vect_length(diff)
    x_grad = -2*diff/(1+diff_square_norm)
    return (x_grad, -x_grad)

def negative_gradient(x, z):
    diff = x-z
    diff_square_norm = square_vect_length(diff)
    x_grad = gamma*2*diff/(0.001+diff_square_norm*(1+diff_square_norm))
    x_grad = np.clip(x_grad, -1, 1)
    return (x_grad, -x_grad)

def update(embs, positions, updates):
    embs[positions, :] -= updates

for batch_no in range(num_batches):
    if np.any(np.isnan(embeddings)):
        print "found nans at batch", batch_no
        break
    learning_rate = (init_learning_rate*(num_batches-batch_no)/num_batches)/batch_size
    true_batch = np.random.choice(num_links, size=batch_size, p=probabilities)
    
    # for each batch, (x, y)'s are true pairs and (?, z)'s are corruptions
    x_locations = link_data[0][true_batch]
    y_locations = link_data[1][true_batch]
    xs = embeddings[x_locations, :]
    ys = embeddings[y_locations, :]
    x_grads, y_grads = positive_gradient(xs, ys)
    z_grads = []
    corruptions = []

    for corruption_no in range(num_corruptions):
        corrupt_batch = np.random.choice(num_pts, size=batch_size, p=negative_sample_probabilities)
        if not getrandbits(1):
            # corrupt source
            if np.all(corrupt_batch != y_locations):
                corruptions.append(corrupt_batch)
                z_grad, y_corr_grads = negative_gradient(embeddings[corrupt_batch, :], ys)
                z_grads.append(z_grad)
                y_grads += y_corr_grads
                no_skip += 1
        else:
            # corrupt target
            if np.all(corrupt_batch != x_locations):
                x_corr_grads, z_grad = negative_gradient(xs, embeddings[corrupt_batch, :])
                z_grads.append(z_grad)
                x_grads += x_corr_grads
                no_skip += 1
            
    update(embeddings, x_locations, learning_rate * x_grads)
    update(embeddings, y_locations, learning_rate * y_grads)
    for corruption, z_grad in zip(corruptions, z_grads):
        update(embeddings, corruption, learning_rate * z_grad)
    
    if batch_no % 5000 == 0:
        print "Finished batch "+str(batch_no)

print "done"
total = num_corruptions * num_batches
print "Skipped", (total - no_skip), "out of", total

plt.plot(embeddings[:, 0], embeddings[:, 1])
plt.savefig("test_image.png")
