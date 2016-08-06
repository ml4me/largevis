import numpy as np
from scipy.optimize import minimize
from random import getrandbits
import matplotlib.pyplot as plt

# ddtype = 'float64'

# Helper functions.
# input (probably not most efficient way of doing this,
# but it's not meant to last):
# input_embeddings = np.array([[np.sin(x*np.pi*2/num_pts), \
#     np.cos(x*np.pi*2/num_pts)] for x in range(num_pts)])
# def shifted(n):
#     return range(n, num_pts)+range(n)

# ## CIRCLE:
# link_data = [np.array(range(num_pts)*num_nns), \
#     np.array([x for i in range(num_nns) for x in shifted(i+1)])]
# link_distances = np.concatenate([(1+i)*np.ones(num_pts) \
#     for i in range(num_nns)])

# ## LINE:
# link_data = [np.array(range(num_pts-1)), np.array(range(1, num_pts))]
# link_distances = np.ones(num_pts-1)

# ## MNIST SAMPLE:
# import pickle
# p1, p2, link_distances = pickle.load(
# open("/home/temerick/mnist_subset_data.pkl", "r"))
# p1.dtype = ddtype
# p2.dtype = ddtype
# link_distances.dtype = ddtype
# link_data = [p1, p2]


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


def calculate_pjis(xis, yis, dists, perplexity=50, tolerance=0.001):
    max_xi = int(np.max(xis))
    sigmas = np.zeros(max_xi+1)
    pjis = np.zeros_like(xis, dtype='float')
    for xi in range(max_xi+1):
        matches = (xis == xi)
        sigmas[xi], pjis[matches] = \
            calculate_sigma_and_pjis_for_fixed_xi(np.power(dists[matches], 2),
                                                  perplexity, tolerance)
    return pjis


def wijs_from(pijs, xis, yis, N):
    from scipy import sparse
    half_wijs = sparse.coo_matrix((pijs, (xis, yis)), shape=(N, N))
    return (half_wijs.tocsc() + half_wijs.tocoo().transpose()) / (2 * N)


def pjis_from(sigma, square_dists):
    exps = np.exp(-square_dists / sigma)
    return exps / np.sum(exps)

# init_values = np.array(range(0, 200), dtype=ddtype) / 20


def calculate_sigma_and_pjis_for_fixed_xi(square_dists, perplexity=50,
                                          tolerance=1e-5):
    def to_minimize(s):  # actually learns 2*sigma^2
        exps = np.exp(-square_dists / (2*np.power(s, 2)))
        p_jis = exps / np.sum(exps)
        return np.abs(perplexity + np.sum(np.log2(p_jis)))

    sigma = minimize(to_minimize, 1.0, method="Nelder-Mead",
                     tol=tolerance, options={'maxiter': 200}).x
    #
    # starts = [0.5, 0.05, 0.1, 1.0, 0.001, 10.0, 20.0]
    # start_values = map(to_minimize, starts)
    # sorted_idxs = sorted(range(len(start_values)), key=start_values.__getitem__)
    #
    # for i, idx in enumerate(sorted_idxs):
    #     sigma = minimize(to_minimize, starts[idx], method="Nelder-Mead",
    #                      tol=tolerance, options={'maxiter': 200}).x
    #     if np.any(np.isnan(sigma)):
    #         if i == len(starts) - 1:
    #             print "Found nans in sigma"
    #             import sys
    #             sys.exit()
    #     elif np.any(np.isinf(sigma)):
    #         if i == len(starts) - 1:
    #             print "Found infinities in sigma"
    #             import sys
    #             sys.exit()
    #     else:
    #         break
    # result = np.sqrt(sigma) / 2
    pji_values = pjis_from(sigma, square_dists)
    return sigma, pji_values


def sgd(data, num_batches=50000, batch_size=1, init_learning_rate=1.0,
        gamma=7, num_corruptions=5, output_dim=2, init=None):

    print data.shape
    num_pts = data.shape[0]
    link_data = [data[:, 0], data[:, 1]]
    link_distances = data[:, 2]

    print "max link distance:", np.max(link_distances)
    link_distances /= np.max(link_distances)
    print "max distance:", link_distances.max()
    print "min distance:", link_distances.min()

    no_skip = 0

    # computed once from input:
    pjis = calculate_pjis(link_data[0], link_data[1], link_distances)
    wijs = wijs_from(pjis, link_data[0], link_data[1], num_pts)
    flattened_wijs = wijs[link_data[0], link_data[1]]

    probabilities = softmax(np.array(flattened_wijs).reshape(-1))
    items, counts = np.unique(
        np.concatenate([link_data[0], link_data[1]]),
        return_counts=True
    )
    negative_sample_probabilities = softmax(np.power(counts, 0.75))

    if init is not None:
        embeddings = init
    else:
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
        x_grad = np.clip(x_grad, -5.0, 5.0)
        return (x_grad, -x_grad)

    def update(embs, positions, updates):
        embs[positions, :] -= updates

    for batch_no in range(num_batches):
        if np.any(np.isnan(embeddings)):
            print "found nans at batch", batch_no
            import sys
            sys.exit()
        learning_rate = init_learning_rate*(num_batches-batch_no)/num_batches
        true_batch = \
            np.random.choice(num_links, size=batch_size, p=probabilities)

        # for each batch, (x, y)'s are true pairs and (?, z)'s are corruptions
        x_locations = link_data[0][true_batch]
        y_locations = link_data[1][true_batch]
        xs = embeddings[x_locations, :]
        ys = embeddings[y_locations, :]
        x_grads, y_grads = positive_gradient(xs, ys)
        z_grads = []
        corruptions = []

        for corruption_no in range(num_corruptions):
            corrupt_batch = \
                np.random.choice(num_pts,
                                 size=batch_size,
                                 p=negative_sample_probabilities)
            if not getrandbits(1):
                # corrupt source
                if np.all(corrupt_batch != y_locations):
                    corruptions.append(corrupt_batch)
                    z_grad, y_corr_grads = \
                        negative_gradient(embeddings[corrupt_batch, :], ys)
                    z_grads.append(z_grad)
                    y_grads += y_corr_grads
                    no_skip += 1
            else:
                # corrupt target
                if np.all(corrupt_batch != x_locations):
                    x_corr_grads, z_grad = \
                        negative_gradient(xs, embeddings[corrupt_batch, :])
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
    return embeddings
