"""Embeddings: classical Euclidean MDS and hyperbolic h-MDS (Constitution sec 2.1).

Euclidean MDS (classical Torgerson) recovers points living in R^dim from a
distance matrix with ~zero stress. h-MDS (Sala et al., ICML 2018) exactly
recovers genuinely-hyperbolic distance matrices by working on the hyperboloid
(Gans) model: ``cosh`` of the geodesic distances forms a Gram-like matrix whose
top eigenvectors give the spatial coordinates back to machine precision.
"""
import numpy as np


def euclidean_mds(D, dim):
    """Classical (Torgerson) MDS.

    Parameters
    ----------
    D : (n, n) array_like
        Symmetric Euclidean distance matrix.
    dim : int
        Target embedding dimension.

    Returns
    -------
    coords : (n, dim) ndarray
        Embedded coordinates.
    stress : float
        Normalized stress of the embedding (see :func:`mds_stress`).
    """
    D = np.asarray(D, dtype=float)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)  # ascending eigenvalues
    idx = np.argsort(w)[::-1][:dim]
    L = np.clip(w[idx], 0, None)
    coords = V[:, idx] * np.sqrt(L)
    return coords, mds_stress(D, coords)


def mds_stress(D, coords):
    """Normalized stress between target distances ``D`` and embedded distances.

    Parameters
    ----------
    D : (n, n) array_like
        Target distance matrix.
    coords : (n, d) array_like
        Embedded coordinates.

    Returns
    -------
    float
        ``sqrt( sum_{i<j} (Demb - D)^2 / sum_{i<j} D^2 )``.
    """
    D = np.asarray(D, dtype=float)
    coords = np.asarray(coords, dtype=float)
    Demb = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    iu = np.triu_indices_from(D, 1)
    return float(np.sqrt(np.sum((Demb[iu] - D[iu]) ** 2) / np.sum(D[iu] ** 2)))


def hmds(D, rank):
    """Hyperbolic MDS on the hyperboloid (Gans) model (Sala et al., ICML 2018).

    Recovery is EXACT only when D comes from genuinely-hyperbolic points whose
    intrinsic dimension is <= rank (then the reconstruction matches D to ~machine
    precision). For arbitrary (non-hyperbolic, or higher-rank) data this is a low-rank
    approximation with nonzero distortion — always report avg_distortion / stress
    rather than assuming exact recovery. (Note also that h-MDS is not scale-invariant:
    non-hyperbolic data needs a learned scale tau; see Sala et al. sec 4.)

    Parameters
    ----------
    D : (n, n) array_like
        Hyperbolic geodesic distance matrix.
    rank : int
        Target spatial dimension (number of spatial coordinates).

    Returns
    -------
    Xvec : (n, rank) ndarray
        Recovered spatial (Gans) coordinates.
    """
    D = np.asarray(D, dtype=float)
    Y = np.cosh(D)
    w, V = np.linalg.eigh(-Y)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order][:rank], 0, None)
    return V[:, order][:, :rank] * np.sqrt(w)


def hyperboloid_distances(Xvec):
    """Geodesic distances on the hyperboloid from spatial (Gans) coordinates.

    Parameters
    ----------
    Xvec : (n, rank) array_like
        Spatial coordinates; the time-like coordinate is
        ``x0 = sqrt(1 + ||Xvec||^2)``.

    Returns
    -------
    D : (n, n) ndarray
        Symmetric geodesic distance matrix with zero diagonal.
    """
    Xvec = np.asarray(Xvec, dtype=float)
    x0 = np.sqrt(1 + (Xvec ** 2).sum(1, keepdims=True))
    M = x0 @ x0.T - Xvec @ Xvec.T
    D = np.arccosh(np.clip(M, 1.0, None))
    # A point's geodesic distance to itself is identically zero; clear the
    # tiny arccosh roundoff (~1e-8) that survives the >=1 clip on the diagonal.
    np.fill_diagonal(D, 0.0)
    return D


def avg_distortion(D, Drec):
    """Mean relative distortion over the upper-triangle pairs.

    Parameters
    ----------
    D : (n, n) array_like
        Reference distance matrix.
    Drec : (n, n) array_like
        Reconstructed distance matrix.

    Returns
    -------
    float
        ``mean_{i<j} |Drec - D| / D``.
    """
    D = np.asarray(D, dtype=float)
    Drec = np.asarray(Drec, dtype=float)
    iu = np.triu_indices_from(D, 1)
    return float(np.mean(np.abs(Drec[iu] - D[iu]) / D[iu]))
