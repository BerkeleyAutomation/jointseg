import numpy as np
import matplotlib.pyplot as plt

class D2Descriptor(object):
    """A D2 Descriptor is a transform-invariant shape metric for 3D meshes.

    Essentially, the D2 descriptor is a probability distribution for the
    distance between two uniformly-selected random points on the surface of the
    mesh.

    Note that this D2 descriptor produces two functions based on particular
    scalings of the input mesh. First, it isotropically scales the mesh so its
    first principal eigenvalue is one. This factors out scale differences
    between meshes. Then, it anisotropically scales the mesh so that all of its
    eigenvalues are one. This gives an estimate of the mesh unbiased for
    thinness or thickness along a particular dimension.
    """

    def __init__(self, mesh, n_samples=1024**2, n_verts=128, verts=None, evals=None):
        """Create a D2 shape descriptor for the given mesh.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to create a descriptor of.

        n_samples : int
            The number of samples to use for creating the descriptor.

        n_bins : int
            The number of bins in the shape histogram.

        n_verts : int
            The number of vertices used to represent the piecewise-continuous
            shape descriptor function.

        verts : :obj:`numpy.ndarray` of float, optional
            The vertices of the descriptor. If specified, the descriptor uses
            these vertices and does not compute a new mapping.

        evals : :obj:`numpy.ndarray` of float
            A 3-entry list of the principal eigenvalues for the mesh. If
            specified, we do not recompute the eigenvalues for the mesh.
        """
        self._mesh = mesh
        self._n_samples = n_samples
        self._n_verts = n_verts
        self._orig_evals = evals

        if verts is None:
            self._compute_histogram()
        else:
            self._verts = np.array(verts)

    @property
    def verts(self):
        return self._verts

    def plot(self):
        """Plot the descriptors on a line plot.
        """
        for i in range(self._verts.shape[0]):
            plt.plot(range(self._n_verts), self._verts[i,:])
        plt.show()

    def distance_to(self, other):
        """Compute the distance between two D2 descriptors.

        This distance is equal to the minimum L1 distance between the two
        functional sets for the two mesh scalings in each descriptor, plus some
        cost based on the difference between the original eigenvalues.

        Returns
        -------
        float
            The distance between the two D2 descriptors.

        Raises
        ------
        ValueError
            If the compared descriptors do not have the same number of vertices.
        """
        if self._n_verts != other._n_verts:
            raise ValueError('Compared descriptors must have the same number of vertices.')

        eig_diff = np.linalg.norm(self._orig_evals - other._orig_evals)

        min_dist = np.infty
        for i in range(self._verts.shape[0]):
            delta = self._verts[i] - other._verts[i]
            dist = np.dot(delta, delta)
            if dist < min_dist:
                min_dist = dist
        min_dist = np.sqrt(min_dist**2 + 0.1*eig_diff**2)
        return min_dist

        delta = self._verts[0] - other._verts[0]
        return np.dot(delta, delta)

    def _compute_histogram(self):
        """Compute the vertices for this D2 descriptor via sampling and
        histogram binning.
        """
        self._verts = []

        o1_mesh = self._mesh.copy()
        o2_mesh = self._mesh.copy()
        o3_mesh = self._mesh.copy()
        self._orig_evals = o1_mesh.scale_principal_eigenvalues([1.0])
        o2_mesh.scale_principal_eigenvalues([1.0, 1.0])
        o3_mesh.scale_principal_eigenvalues([1.0, 1.0, 1.0])

        meshes = [o1_mesh, o2_mesh, o3_mesh]

        for m in meshes:
            first_set = self._mesh.random_points(self._n_samples)
            second_set = self._mesh.random_points(self._n_samples)
            dists = []
            for i in range(self._n_samples):
                dist = np.linalg.norm(first_set[i] - second_set[i])
                dists.append(dist)

            hist, bins = np.histogram(dists, bins=self._n_verts)
            s = float(sum(hist))
            self._verts.append(hist / s)

        self._verts = np.array(self._verts)

