import numpy as np
from sklearn.utils.extmath import randomized_svd

class RandomizedLDA:
    def __init__(self, n_components=None, n_oversamples=10, n_iter=2, random_state=None):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)

        # ---- Compute class means ----
        mean_total = X.mean(axis=0)
        means = []
        priors = []

        for c in classes:
            Xc = X[y == c]
            means.append(Xc.mean(axis=0))
            priors.append(len(Xc) / n_samples)

        means = np.vstack(means)
        priors = np.array(priors)

        # ---- Compute within-class scatter S_W ----
        Sw = np.zeros((n_features, n_features))
        for i, c in enumerate(classes):
            Xc = X[y == c] - means[i]
            Sw += Xc.T @ Xc

        # ---- Compute between-class scatter S_B ----
        Sb = np.zeros((n_features, n_features))
        for i in range(n_classes):
            mean_diff = (means[i] - mean_total).reshape(-1, 1)
            Sb += priors[i] * (mean_diff @ mean_diff.T)

        # ---- Whitening transform via randomized SVD on Sw ----
        U, S, Vt = randomized_svd(
            Sw,
            n_components=min(n_features, self.n_components + self.n_oversamples),
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        # avoid division by zero
        S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-12))

        # whitening matrix
        W = U @ S_inv_sqrt

        # ---- Transform Sb into whitened space ----
        Sb_tilde = W.T @ Sb @ W

        # ---- Solve eigenproblem via randomized SVD ----
        U2, S2, Vt2 = randomized_svd(
            Sb_tilde,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        # final projection matrix
        self.scalings_ = W @ U2[:, :self.n_components]

        # store means for transform
        self.xbar_ = mean_total

        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.xbar_) @ self.scalings_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)