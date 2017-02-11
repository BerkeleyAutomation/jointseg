import numpy as np
import cvxopt as cvx
from scipy.linalg import lu

from meshpy import OffFile
from visualization import Visualizer3D

from segmentation import Shape
from color_picker import indexed_color

def create_shape_library(mesh_files):
    shapes = []
    for mesh_file in mesh_files:
        mesh = OffFile(mesh_file).read()
        shapes.append(Shape(mesh, 50))

    for shape in shapes:
        for i in range(2, 8):
            shape.generate_random_segmentations(1, i)

    distances = []
    segment_map = {}

    # Compute minimal distance between shapes and repetition counts
    for shape in shapes:
        for segment in shape.segments:
            segment_map[segment] = ([],[],[]) # Min distances and corresponding counts
            for other_shape in shapes:

                # If same shape, min distance is 0.0 and count is same count
                if other_shape == shape:
                    segment_map[segment][0].append(0.0)
                    segment_map[segment][1].append(segment.repetition_count)
                    segment_map[segment][2].append(segment.ncut_cost)
                    continue

                # Otherwise, find closest segment in other shape
                min_distance = float('inf')
                repetiton_count = 0.0
                ncut_cost = 0.0
                for other_segment in other_shape.segments:
                    distance = segment.distance_to(other_segment)

                    # TEMP
                    #color = indexed_color(0)
                    #Visualizer3D.mesh(segment.mesh, style='surface', color=color)
                    #color = indexed_color(1)
                    #Visualizer3D.mesh(other_segment.mesh, style='surface', color=color)
                    #print distance
                    #Visualizer3D.show()
                    #segment.d2_descriptor.plot()
                    #other_segment.d2_descriptor.plot()

                    if distance < min_distance:
                        min_distance = distance
                        repetition_count = other_segment.repetition_count
                        ncut_cost = other_segment.ncut_cost
                segment_map[segment][0].append(min_distance)
                segment_map[segment][1].append(repetition_count)
                segment_map[segment][2].append(ncut_cost)
                distances.append(min_distance)

    # Add weights to individual segments
    sigma = 2*np.median(distances)**2
    for segment in segment_map:
        weight = 0.0
        dists = segment_map[segment][0]
        counts = segment_map[segment][1]
        costs = segment_map[segment][2]
        for dist, count, cost in zip(dists, counts, costs):
            weight += np.exp(-1*dist**2/sigma) * count / cost
        segment.weight = weight

    #for shape in shapes:
    #    print "SHAPE"
    #    print "="*60
    #    for segment in shape.segments:
    #        print "Weight: {}".format(segment.weight * segment.area / shape.mesh.surface_area())
    #        print segment_map[segment][0]
    #        print segment_map[segment][1]
    #        Visualizer3D.mesh(segment.mesh, style='surface', color=(0.5, 0.5, 0.3))
    #        Visualizer3D.show()

    return shapes, sigma

def create_C(shape1, shape2, sigma):
    weights = []

    # Add w_seg weights for individual segment indicators
    for shape in [shape1, shape2]:
        for segment in shape.segments:
            w_seg = segment.weight * segment.area / shape.mesh.surface_area()
            weights.append(w_seg)

    # Add w_corr weights for pairwise individual segment indicators
    for s1 in shape1.segments:
        for s2 in shape2.segments:
            dist = s1.distance_to(s2)
            w = np.exp(-1*dist**2/sigma) * s1.area / shape1.mesh.surface_area()
            weights.append(w)
    for s1 in shape2.segments:
        for s2 in shape1.segments:
            dist = s1.distance_to(s2)
            w = np.exp(-1*dist**2/sigma) * s1.area / shape2.mesh.surface_area()
            weights.append(w)

    weights = np.array(weights)
    return -1.0*weights

def create_AB(shape1, shape2):
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

def create_GH(shape1, shape2):
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

def run_optimization(shape1, shape2, sigma):

    A, B = create_AB(shape1, shape2)
    G, H = create_GH(shape1, shape2)
    C    = create_C(shape1, shape2, sigma)

    # Solve the convex optimization problem
    np.savetxt('A.txt', A, fmt='%d')
    np.savetxt('B.txt', B, fmt='%d')
    np.savetxt('G.txt', G, fmt='%d')
    np.savetxt('H.txt', H, fmt='%d')
    np.savetxt('C.txt', C, fmt='%.3e')
    A = cvx.matrix(A)
    B = cvx.matrix(B)
    G = cvx.matrix(G)
    H = cvx.matrix(H)
    C = cvx.matrix(C)

    sol = cvx.solvers.lp(C,G,H,A=A,b=B,solver='glpk')
    x = np.squeeze(np.array(sol['x']))

    print x
    # Greedily select the highest-ranked segments
    start_ind = 0
    segmentations = []
    for shape in [shape1, shape2]:
        print "SHAPE"
        final_segments = []
        segs_scores = zip(shape.segments, x[start_ind : start_ind + shape.n_segments])
        while(len(segs_scores) > 0):
            final_seg_score = max(segs_scores, key=lambda x : x[1])
            final_seg = final_seg_score[0]
            final_segments.append(final_seg)
            segs_scores.remove(final_seg_score)
            new_segs_scores = []
            for seg_score in segs_scores:
                seg = seg_score[0]
                if not seg.intersects(final_seg):
                    new_segs_scores.append(seg_score)
            segs_scores = new_segs_scores
        segmentations.append(final_segments)
        start_ind += shape.n_segments

    for segmentation in segmentations:
        for i, segment in enumerate(segmentation):
            Visualizer3D.mesh(segment.mesh, style='surface', color=indexed_color(i))
        Visualizer3D.show()

def main():
    filenames = ['./meshes/glasses/{}.off'.format(i) for i in range(43, 48)]

    shapes, sigma = create_shape_library(filenames)
    run_optimization(shapes[0], shapes[1], sigma)


if __name__ == '__main__':
    main()

