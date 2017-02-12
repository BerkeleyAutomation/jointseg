import numpy as np
import cvxopt as cvx
from sympy import Matrix
import scipy.sparse as sp

from meshpy import OffFile
from visualization import Visualizer3D

from segmentation import Shape
from color_picker import indexed_color

cvx.solvers.options['show_progress'] = False

def li_matrix(A):
    rref, rowinds = Matrix(A.T).rref()
    rows = []
    for ind in rowinds:
        rows.append(A[ind])
    return np.array(rows)

class PairwiseJointSeg(object):
    """The result of a joint shape optimization.
    """

    def __init__(self, shapes, A, B, G, H, C):
        self._C = C
        self._shapes = shapes
        self._seg_indicators = []
        self._seg_weights = []
        self._map_indicators = []
        self._map_weights = []
        self._segmentations = []

        A = cvx.matrix(A)
        B = cvx.matrix(B)
        G = cvx.matrix(G)
        H = cvx.matrix(H)
        C = cvx.matrix(C)

        sol = cvx.solvers.lp(C,G,H,A=A,b=B,solver='glpk')
        x = np.squeeze(np.array(sol['x']))
        self._compute_arrs(x)
        self._round_seginds()
        self._compute_segmentations()

    @property
    def segmentations(self):
        return self._segmentations

    @property
    def similarity_score(self):
        score = 0.0
        for map_inds, map_weights in zip(self._map_indicators, self._map_weights):
            score += map_inds.T.dot(map_weights)
        return score

    def _compute_segmentations(self):
        self._segmentations = []
        for shape, seginds in zip(self._shapes, self._seg_indicators):
            segments = []
            for i, segment in enumerate(shape.segments):
                if seginds[i] == 1:
                    segments.append(segment)
            self._segmentations.append(segments)

    def _round_seginds(self):
        for i, shape in enumerate(self._shapes):
            seginds = self._seg_indicators[i]
            segset = zip(shape.segments, [x for x in range(shape.n_segments)], seginds)
            while len(segset) > 0:
                max_segtup = max(segset, key=lambda x : x[2])
                seg = max_segtup[0]
                index = max_segtup[1]
                seginds[index] = 1

                new_segset = []
                for segtup in segset:
                    other_seg = segtup[0]
                    other_index = segtup[1]
                    if other_seg == seg:
                        continue
                    elif other_seg.intersects(seg):
                        seginds[other_index] = 0.0
                    else:
                        new_segset.append(segtup)
                segset = new_segset

    def _compute_arrs(self, x):
        index = 0
        shape1 = self._shapes[0]
        shape2 = self._shapes[1]
        crossterms = shape1.n_segments * shape2.n_segments

        self._seg_indicators.append(np.array(x[index : index + shape1.n_segments]))
        self._seg_weights.append(np.array(self._C[index : index + shape1.n_segments]))
        index += shape1.n_segments

        self._seg_indicators.append(np.array(x[index : index + shape2.n_segments]))
        self._seg_weights.append(np.array(self._C[index : index + shape2.n_segments]))
        index += shape2.n_segments

        self._map_indicators.append(np.array(x[index : index + crossterms]))
        self._map_weights.append(np.array(self._C[index : index + crossterms]))
        index += crossterms

        self._map_indicators.append(np.array(x[index : index + crossterms]))
        self._map_weights.append(np.array(self._C[index : index + crossterms]))

class ShapeInfo(object):
    """Persistent data for segmenting a shape.
    """

    def __init__(self, shape, shapes, seg_to_seg_dists, sigma):
        """TODO
        """
        self._shape = shape
        self._xi = np.zeros(shape.n_segments)
        self._xijs = {s : np.zeros(shape.n_segments) for s in shapes if s !=shape}
        self._yijs = {s : np.zeros(shape.n_segments * s.n_segments) for s in shapes if s != shape}
        self._wseg = self._compute_wseg()
        self._wcorrs = self._compute_wcorrs(shapes, seg_to_seg_dists, sigma)

    @property
    def wseg(self):
        return self._wseg

    @property
    def wcorrs(self):
        return self._wcorrs

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, new_xi):
        self._xi = np.array(new_xi)

    @property
    def xijs(self):
        return self._xijs

    @xijs.setter
    def xijs(self, xij):
        self._xijs = new_xij

    @property
    def yijs(self):
        return self._yijs

    @yijs.setter
    def yijs(self, yij):
        self._yijs = yij

    def _compute_wseg(self):
        weights = []
        for seg in self._shape.segments:
            w_seg = seg.weight * seg.area / self._shape.mesh.surface_area()
            weights.append(w_seg)
        return np.array(weights)

    def _compute_wcorrs(self, shapes, seg_to_seg_dists, sigma):
        wcorrs = {}
        for other_shape in shapes:
            if other_shape == self._shape:
                continue
            weights = []
            for seg1 in self._shape.segments:
                for seg2 in other_shape.segments:
                    dist = seg_to_seg_dists[seg1][seg2]
                    w = np.exp(-1*dist**2/sigma) * seg1.area / self._shape.mesh.surface_area()
                    weights.append(w)
            wcorrs[other_shape] = np.array(weights)
        return wcorrs

class ShapeLibrary(object):
    """ASDF
    """

    def __init__(self, meshes, n_patches=25, n_segmentations=1,
            max_segs=10, max_segs_per_shape=200):
        """ASDF
        """

        self._meshes = meshes
        self._n_patches = n_patches
        self._n_segmentations = n_segmentations
        self._shapes = []
        self._shape_info = {}

        # Create shapes
        for mesh in self._meshes:
            self._shapes.append(Shape(mesh, self._n_patches))

        # Create segmentations for each shape
        for shape in self._shapes:
            for i in range(2, max_segs+1):
                shape.generate_random_segmentations(self._n_segmentations, i)

        # Maps segments to dicts with keys of other segments and values of
        # distance
        seg_to_seg_dists = {}

        # Maps segments to a dict of shapes, with values being the closest
        # segment in that shape and the distance between the segments.
        seg_to_min_dists = {}

        # Initialize maps
        for shape in self._shapes:
            for seg in shape.segments:
                seg_to_seg_dists[seg] = {}
                seg_to_min_dists[seg] = {}

                seg_to_min_dists[seg][shape] = (seg, 0.0)

        # Compute distances between shapes
        for i, shape1 in enumerate(self._shapes):
            for seg1 in shape1.segments:
                for shape2 in self._shapes[i+1:]:
                    seg_to_min_dists[seg1][shape2] = (None, float('inf'))

                    for seg2 in shape2.segments:
                        if shape1 not in seg_to_min_dists[seg2]:
                            seg_to_min_dists[seg2][shape1] = (None, float('inf'))

                        dist = seg1.distance_to(seg2)
                        seg_to_seg_dists[seg1][seg2] = dist
                        seg_to_seg_dists[seg2][seg1] = dist
                        if dist < seg_to_min_dists[seg1][shape2][1]:
                            seg_to_min_dists[seg1][shape2] = (seg2, dist)
                        if dist < seg_to_min_dists[seg2][shape1][1]:
                            seg_to_min_dists[seg2][shape1] = (seg1, dist)

        # Set of all computed distances
        distances = []

        # Compute sigma
        for seg in seg_to_min_dists:
            for shape, tup in seg_to_min_dists[seg].iteritems():
                dist = tup[1]
                if dist != 0.0:
                    distances.append(dist)
        sigma = 2*np.median(distances)**2

        # Compute weights for each segment
        for seg in seg_to_min_dists:
            seg.weight = 0.0
            for shape, tup in seg_to_min_dists[seg].iteritems():
                dist = tup[1]
                rep_count = tup[0].repetition_count
                weight = np.exp(-1*dist**2/sigma)
                seg.weight += weight

        # Prune each shape's segments, keeping the top max_segs_per_shape plus
        # the full segmentation with the highest minimum segment weight
        for shape in self._shapes:
            shape.prune_to(max_segs_per_shape)

        # Re-create the seg_to_seg_dists mapping
        new_seg_to_seg_dists = {}
        for shape in self._shapes:
            print len(shape.segments)
            for seg in shape.segments:
                if seg not in new_seg_to_seg_dists:
                    new_seg_to_seg_dists[seg] = {}
                for other_shape in self._shapes:
                    if shape != other_shape:
                        for other_seg in other_shape.segments:
                            new_seg_to_seg_dists[seg][other_seg] = seg_to_seg_dists[seg][other_seg]

        self._seg_to_seg_dists = new_seg_to_seg_dists
        self._sigma = sigma

        for shape in self._shapes:
            self._shape_info[shape] = ShapeInfo(shape, self._shapes,
                                                self._seg_to_seg_dists, self._sigma)

    @property
    def shapes(self):
        return self._shapes

    def pairwise_joint_segmentation(self, shape1, shape2):
        A, B = self._create_AB(shape1, shape2)
        G, H = self._create_GH(shape1, shape2)
        C    = self._create_C(shape1, shape2)

        pjs = PairwiseJointSeg([shape1, shape2], A, B, G, H, C)

        segmentations = pjs.segmentations
        for segmentation in segmentations:
            for i, segment in enumerate(segmentation):
                Visualizer3D.mesh(segment.mesh, style='surface', color=indexed_color(i))
            Visualizer3D.show()

    def _run_ss_opt(self, shape, gamma):
        """Run a single shape segmentation optimization step.
        """
        shape_info = self._shape_info[shape]
        wseg = shape_info.wseg
        wsquare = np.zeros(wseg.shape)
        for xij in shape_info.xijs.itervalues():
            wsquare += 2*xij
        wsquare *= gamma * len(self.shapes) / (len(self.shapes))**2
        Q = -1.0 * (wseg + wsquare)
        A = np.zeros((shape.n_patches, shape.n_segments))
        for i, patch in enumerate(shape.patches):
            for j, segment in enumerate(shape.segments):
                if patch in segment.patches:
                    A[i][j] = 1.0
        A = li_matrix(A)
        B = np.ones((A.shape[0], 1))
        G = np.vstack((-1.0*np.eye(shape.n_segments), np.eye(shape.n_segments)))
        H = np.vstack((np.zeros((shape.n_segments, 1)), np.ones((shape.n_segments, 1))))
        x = -1.0*gamma*len(self.shapes) / (len(self.shapes)**2)*2
        P = -1.0 * x * np.eye(shape.n_segments)
        Q = cvx.matrix(Q)
        A = cvx.matrix(A)
        B = cvx.matrix(B)
        G = cvx.matrix(G)
        H = cvx.matrix(H)
        P = cvx.matrix(P)
        sol = cvx.solvers.qp(P,Q,G,H,A,B,solver='glpk')
        x = np.squeeze(np.array(sol['x']))
        shape_info.xi = x
        self.round_xi(shape)
        print shape_info.xi

    def _init_xijs(self):
        for shape in self._shapes:
            si = self._shape_info[shape]
            xi = si.xi
            for os in si.xijs.keys():
                si.xijs[os] = xi.copy()

    def _run_joint_opt(self, shape1, shape2, gamma):
        sinfo1 = self._shape_info[shape1]
        sinfo2 = self._shape_info[shape2]

        wcorrs = sinfo1.wcorrs[shape2]
        xi = sinfo1.xi


        A, B = self._create_AB2(shape1, shape2)
        G, H = self._create_GH2(shape1, shape2)

        Q = cvx.matrix(-1.0 * np.hstack((2*gamma*xi, wcorrs)))
        qs = xi.shape[0] + wcorrs.shape[0]

        px = [gamma for i in range(shape1.n_segments)]
        pr = range(shape1.n_segments)
        pc = range(shape1.n_segments)
        P = cvx.spmatrix(px, pr, pc, size=(qs, qs))

        sol = cvx.solvers.qp(P,Q,G,H,A,B, solver='glpk')
        x = np.squeeze(np.array(sol['x']))

        sinfo1.xijs[shape2] = x[0:shape1.n_segments]
        sinfo1.yijs[shape2] = x[shape1.n_segments:]
        self.round_xijs(shape1, shape2)

    def round_xi(self, shape):
        si = self._shape_info[shape]
        xi = si.xi
        segset = zip(shape.segments, [x for x in range(shape.n_segments)], xi)
        while len(segset) > 0:
            max_segtup = max(segset, key=lambda x : x[2])
            seg = max_segtup[0]
            index = max_segtup[1]
            xi[index] = 1.0

            new_segset = []
            for segtup in segset:
                other_seg = segtup[0]
                other_index = segtup[1]
                if other_seg == seg:
                    continue
                elif other_seg.intersects(seg):
                    xi[other_index] = 0.0
                else:
                    new_segset.append(segtup)
            segset = new_segset

    def round_xijs(self, shape1, shape2):
        si = self._shape_info[shape1]
        xij = si.xijs[shape2]
        segset = zip(shape1.segments, [x for x in range(shape1.n_segments)], xij)
        while len(segset) > 0:
            max_segtup = max(segset, key=lambda x : x[2])
            seg = max_segtup[0]
            index = max_segtup[1]
            xij[index] = 1.0

            new_segset = []
            for segtup in segset:
                other_seg = segtup[0]
                other_index = segtup[1]
                if other_seg == seg:
                    continue
                elif other_seg.intersects(seg):
                    xij[other_index] = 0.0
                else:
                    new_segset.append(segtup)
            segset = new_segset
        si.xijs[shape2] = xij

    def _round_and_show(self):
        for shape in self.shapes:
            si = self._shape_info[shape]
            xi = si.xi
            segset = zip(shape.segments, [x for x in range(shape.n_segments)], xi)
            while len(segset) > 0:
                max_segtup = max(segset, key=lambda x : x[2])
                seg = max_segtup[0]
                index = max_segtup[1]
                xi[index] = 1

                new_segset = []
                for segtup in segset:
                    other_seg = segtup[0]
                    other_index = segtup[1]
                    if other_seg == seg:
                        continue
                    elif other_seg.intersects(seg):
                        xi[other_index] = 0.0
                    else:
                        new_segset.append(segtup)
                segset = new_segset

            for i, segment in enumerate(shape.segments):
                if xi[i] == 1.0:
                    Visualizer3D.mesh(segment.mesh, style='surface', color=indexed_color(i))
            Visualizer3D.show()

    def _create_C(self, shape1, shape2):
        wseg1 = self._shape_info[shape1].wseg
        wseg2 = self._shape_info[shape2].wseg
        wcoors12 = self._shape_info[shape1].wcorrs[shape2]
        wcoors21 = self._shape_info[shape2].wcorrs[shape1]
        weights = np.hstack((wseg1, wseg2, wcoors12, wcoors21))
        return -1.0*weights

    def _create_AB(self, shape1, shape2):
        A_matrices = []

        # Create A matrix blocks for each shape
        for shape in [shape1, shape2]:
            A = np.zeros((shape.n_patches, shape.n_segments))
            for i, patch in enumerate(shape.patches):
                for j, segment in enumerate(shape.segments):
                    if patch in segment.patches:
                        A[i][j] = 1.0
            A_matrices.append(A)

        cross_terms = 2*shape1.n_segments*shape2.n_segments
        zeros_top = np.zeros((shape1.n_patches, cross_terms))
        zeros_bot = np.zeros((shape2.n_patches, cross_terms))
        A_top = np.hstack((A_matrices[0], np.zeros((shape1.n_patches, shape2.n_segments)), zeros_top))
        A_bot = np.hstack((np.zeros((shape2.n_patches, shape1.n_segments)), A_matrices[1], zeros_bot))
        A = np.vstack((A_top, A_bot))
        A = np.vstack(set(tuple(row) for row in A))

        B = np.ones((A.shape[0], 1))

        return A, B

    def _create_AB2(self, shape1, shape2):
        """Asdf
        """
        A = np.zeros((shape1.n_patches, shape1.n_segments))
        for i, patch in enumerate(shape1.patches):
            for j, segment in enumerate(shape1.segments):
                if patch in segment.patches:
                    A[i][j] = 1.0
        A = li_matrix(A)

        ax = []
        ar = []
        ac = []

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i][j] == 1.0:
                    ax.append(1.0)
                    ar.append(i)
                    ac.append(j)

        aheight = A.shape[0]
        A = cvx.spmatrix(ax, ar, ac, size=(aheight, shape1.n_segments + shape1.n_segments * shape2.n_segments))
        B = cvx.matrix(np.ones((aheight, 1)))

        return A, B

    def _create_GH2(self, shape1, shape2):
        sx1 = shape1.n_segments
        sx2 = shape2.n_segments
        sy = sx1*sx2
        xji = self._shape_info[shape2].xijs[shape1]

        # G is of width (sx1 + sxy), height (2*sx1 + 2*sy + sx1 + sy)
        gx = []
        gr = []
        gc = []

        rs = 0 # row start value

        # Create constraint for 0 <= x_ij <= 1
        for value in [-1.0, 1.0]:
            for i in range(sx1):
                gx.append(value)
                gr.append(rs + i)
                gc.append(i)
            rs += sx1

        # Create constraint for 0 <= y_ij <= 1
        for value in [-1.0, 1.0]:
            for i in range(sy):
                gx.append(value)
                gr.append(rs + i)
                gc.append(sx1 + i)
            rs += sy

        # Create constraint (sum over s') y_ij(s, s') <= x_ij_s
        for i in range(sx1):
            gx.append(-1.0)
            gr.append(rs + i)
            gc.append(i)
            for j in range(sx2*i, sx2*(i+1)):
                gx.append(1.0)
                gr.append(rs + i)
                gc.append(sx1 + j)
        rs += sx1

        # Create constraint sum
        for i in range(sy):
            gx.append(1.0)
            gr.append(rs + i)
            gc.append(sx1 + i)

        G = cvx.spmatrix(gx, gr, gc, size=(3*sx1 + 3*sy, sx1 + sy))

        xjicol = np.zeros((sy, 1))
        for i in range(sy):
            ind = i % sx2
            xjicol[i][0] = xji[ind]

        H = np.vstack((
            np.zeros((sx1, 1)),
            np.ones((sx1, 1)),
            np.zeros((sy, 1)),
            np.ones((sy, 1)),
            np.zeros((sx1, 1)),
            xjicol
        ))

        H = cvx.matrix(H)

        return G, H

    def _create_GH(self, shape1, shape2):
        sx1 = shape1.n_segments
        sx2 = shape2.n_segments
        sy = sx1*sx2

        rowx1 = np.hstack((
            np.eye(sx1), np.zeros((sx1, sx2)), np.zeros((sx1, 2*sy))
        ))
        rowx2 = np.hstack((
            np.zeros((sx2, sx1)), np.eye(sx2), np.zeros((sx2, 2*sy))
        ))
        rowy12 = np.hstack((
            np.zeros((sy, sx1)), np.zeros((sy, sx2)), np.eye(sy), np.zeros((sy, sy))
        ))
        rowy21 = np.hstack((
            np.zeros((sy, sx1)), np.zeros((sy, sx2)), np.zeros((sy, sy)), np.eye(sy)
        ))

        blocky12 = np.zeros((sx1, sy))
        start_pt = 0
        for i in range(sx1):
            for j in range(start_pt, start_pt + sx2):
                blocky12[i][j] = 1
            start_pt += sx2

        blocky21 = np.zeros((sx2, sy))
        start_pt = 0
        for i in range(sx2):
            for j in range(start_pt, start_pt + sx1):
                blocky21[i][j] = 1
            start_pt += sx1

        endy12 = np.zeros((sy, sx2))
        for i in range(sy):
            j = i % sx2
            endy12[i][j] = 1

        endy21 = np.zeros((sy, sx1))
        for i in range(sy):
            j = i % sx1
            endy21[i][j] = 1

        rowblk12 = np.hstack((
            -1*np.eye(sx1), np.zeros((sx1, sx2)), blocky12, np.zeros((sx1, sy))
        ))

        rowblk21 = np.hstack((
            np.zeros((sx2, sx1)), -1*np.eye(sx2), np.zeros((sx2, sy)), blocky21
        ))

        rowend12 = np.hstack((
            np.zeros((sy, sx1)), -1*endy12, np.eye(sy), np.zeros((sy, sy))
        ))

        rowend21 = np.hstack((
            -1*endy21, np.zeros((sy, sx2)), np.zeros((sy, sy)), np.eye(sy)
        ))

        G = np.vstack((
            -1*rowx1,
            rowx1,
            -1*rowx2,
            rowx2,
            -1*rowy12,
            rowy12,
            -1*rowy21,
            rowy21,
            rowblk12,
            rowblk21,
            rowend12,
            rowend21
        ))

        H = np.vstack((
            np.zeros((sx1, 1)),
            np.ones((sx1, 1)),
            np.zeros((sx2, 1)),
            np.ones((sx2, 1)),
            np.zeros((sy, 1)),
            np.ones((sy, 1)),
            np.zeros((sy, 1)),
            np.ones((sy, 1)),
            np.zeros((sx1, 1)),
            np.zeros((sx2, 1)),
            np.zeros((sy, 1)),
            np.zeros((sy, 1))
        ))

        return G, H

def main():
    filenames = ['./meshes/cup/{}.off'.format(i) for i in range(21, 31)]
    meshes = [OffFile(f).read() for f in filenames]

    sl = ShapeLibrary(meshes, n_segmentations=1)
    #sl.pairwise_joint_segmentation(sl.shapes[0], sl.shapes[1])

    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    for s in sl.shapes:
        sl._run_ss_opt(s, 0.0001)
    sl._init_xijs()
    for gamma in gammas:
        for s in sl.shapes:
            sl._run_ss_opt(s, gamma)
        for s in sl.shapes:
            for os in sl.shapes:
                if s != os:
                    sl._run_joint_opt(s, os, gamma)

    sl._round_and_show()

if __name__ == '__main__':
    main()

