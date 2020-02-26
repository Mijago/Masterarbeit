import time

import numpy as np


def sum(a, b=0):
    return np.array([np.sum(a, b)])


def correctMatrix(D, correction='clip'):
    now = time.time()
    # dissimilarities
    N = D.shape[0]

    print("A", time.time() - now)
    PInvDmm = np.linalg.pinv(D)
    print("B", time.time() - now)
    S_mm = -.5 * (
            D
            - 1 / N * np.ones((N, 1)) * sum(D, 0)
            - 1 / N * sum(D, 0).T @ np.ones((1, N))
            + 1 / pow(N, 2) * np.sum(sum(D @ PInvDmm, 0) @ D.T)
    )
    print("C", time.time() - now)
    # Eigval correction
    S_mm = .5 * (S_mm + S_mm.T)

    print("D", time.time() - now)
    tmp = sum(D @ PInvDmm, 0)
    print("E", time.time() - now)
    center_estimates = tmp / N
    print("F", time.time() - now)
    overall_mean_dissimilarity = np.sum(tmp @ D.T) / pow(N, 2)
    print("G", time.time() - now)
    mean_dissimilarity = sum(D, 0) / N
    print("H", time.time() - now)

    w, v = np.linalg.eigh(S_mm)
    print("I", time.time() - now)
    v = np.flip(v, 1)
    w = w[::-1]
    if correction == 'clip':
        w[w < 0] = 0  # clip (currently disabled)
    elif correction == 'flip':
        w[w < 0] = np.abs(w[w < 0])  # clip (currently disabled)
    w = np.diag(w)
    print("J", time.time() - now)

    t_ = (w @ v.T)
    print("K1", time.time() - now)
    S_nm_ = v @ t_
    print("K", time.time()-now)
    S_mm_ = S_nm_
    S_mm_ = 0.5 * (S_mm_ + S_mm_.T)
    print("L", time.time() - now)

    S_nm_pinv = np.linalg.pinv(S_mm_, 1e-4)
    print("M", time.time() - now)
    P = S_nm_pinv @ S_nm_

    def fDis2K(DX_km):
        _N, _m = DX_km.shape
        Cnm = -.5 * (
                DX_km
                - np.ones((_N, 1)) * mean_dissimilarity
                - (center_estimates @ DX_km.T).T * np.ones((1, _m))
                + overall_mean_dissimilarity
        )
        return Cnm @ P, Cnm

    def fDis2D(DX_km, ptId=None):
        Snm, Cnm = fDis2K(DX_km)
        N, m = Snm.shape
        corrected_self_similarities = np.sum((S_nm_pinv @ Snm.T) * Snm.T, 0)

        if ptId is None:
            distances = np.array(
                [[
                    S_nm_[col, col] + corrected_self_similarities[row] - 2 * Snm[row, col]
                    for col in range(0, m)
                ] for row in range(0, N)]
            )
            return distances
        else:
            distances = S_nm_[ptId, ptId] + corrected_self_similarities[0] - 2 * Snm[0, ptId]
            return distances

    return S_nm_, fDis2K, fDis2D
