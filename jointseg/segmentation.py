"""
File for processing random mesh segmentations.
Author: Matthew Matl
"""
import heapq

import numpy as np
import networkx as nx

from visualization import Visualizer3D
from meshpy import Mesh3D

from d2_descriptor import D2Descriptor
from color_picker import indexed_color

class FaceNode(object):
    """A single triangle in a 3D mesh. This class is used
    as the node element in a FaceGraph.
    """

    def __init__(self, index, vertex_indices, vertex_coord_array):
        """Create a FaceNode.

        Parameters
        ----------
        index : int
            An index for this face.

        vertex_indices : `numpy.ndarray` of int
            The indices of the three vertices of this face.
            These will be used to index vertex_coord_array to extract vertex
            coordinates.

        vertex_coords_array : `numpy.ndarray` of float
            An array of 3D vertex coordinates (indexed by vertex_indices).

        Raises
        ------
        ValueError
            If the input data is not of the correct shape.
        """
        if len(vertex_indices) != 3 or vertex_coord_array.shape[1] != 3:
            raise ValueError("A face requires three 3D vertices.")

        self._index = index
        self._vertex_inds = vertex_indices
        self._vertex_coords = vertex_coord_array[vertex_indices]
        self._area = self._compute_area()
        self._normal = self._compute_normal()
        self._edges = self._compute_edges()

    @property
    def index(self):
        """int : The index for this face.
        """
        return self._index

    @property
    def vertex_inds(self):
        """:obj:`numpy.ndarray` of int : The indices for the vertices in this face.
        """
        return self._vertex_inds

    @property
    def vertex_coords(self):
        """:obj:`numpy.ndarray` of float : The three 3D vertex coordinates for
        this face.
        """
        return self._vertex_coords

    @property
    def area(self):
        """float : The area of this face.
        """
        return self._area

    @property
    def normal(self):
        """:obj:`numpy.ndarray` of float : The normalized 3D vector that is
        normal to this face.
        """
        return self._normal

    @property
    def edges(self):
        """:obj:`list` of :obj:`tuple` of int : A list of three ordered tuples
        that contain the indices of vertices that form the endpoints of each
        edge of the face. The first entry in each tuple is always the smaller of
        the two indices in the tuple.
        """
        return self._edges

    def coord_from_index(self, vert_index):
        """Maps a global vertex index to a 3D coordinate.

        Parameters
        ----------
        vert_index : int
            A global vertex index.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3D coordinate corresponding to the given index.
        """
        for i, index in enumerate(self.vertex_inds):
            if index == vert_index:
                return self.vertex_coords[i]
        return None

    def angle_to(self, other):
        """Compute the exterior dihedral angle between this face and an adjacent
        one.

        Parameters
        ----------
        other : :obj:`FaceNode`
            Another face that is adjacent to this one.

        Returns
        -------
        float
            The external dihedral angle between the two faces in radians.
        """
        # Find the coordinates of the two unshared vertex indices
        all_vertex_inds = set(self.vertex_inds) | set(other.vertex_inds)
        if len(all_vertex_inds) != 4:
            raise ValueError('Faces have same vertices -- possible duplicate face.')

        vs = None
        vo = None
        for ind in all_vertex_inds:
            if ind not in self.vertex_inds:
                vo = other.coord_from_index(ind)
            elif ind not in other.vertex_inds:
                vs = self.coord_from_index(ind)

        # Determine if the edge is convex or concave. If the angle between my
        # normal and the line from vs to vo is greater than 90 degrees, the
        # edge is a ridge
        concave = True
        if (vo - vs).dot(self.normal) <= 0:
            concave = False

        # Determine the angle a between the two triangular planes (same as angle
        # between normal vectors).
        # If the exterior angle is concave, then the exterior angle is 180 - a.
        # If the exterior angle is convex, then the exterior angle is 180 + a.
        angle = np.arccos(self.normal.dot(other.normal))
        if concave:
            return np.pi - angle
        else:
            return np.pi + angle 

    def edge_length(self, other):
        """Find the length of the edge between this face and another one.

        Parameters
        ----------
        other : :obj:`FaceNode`
            Another face that is adjacent to this one.

        Returns
        -------
        float
            The length of the edge between the two faces.
        """
        # Find the coordinates of the two unshared vertex indices
        edge = list(set(self.vertex_inds) & set(other.vertex_inds))
        return np.linalg.norm(self.coord_from_index(edge[0]) - (self.coord_from_index(edge[1])))

    def cut_cost(self, other):
        """Return the cut cost for removing the edge between this face and
        another one.

        This is calculated as edge length times min((theta / pi)^10, 1).

        Parameters
        ----------
        other : :obj:`FaceNode`
            Another face that is adjacent to this one.

        Returns
        -------
        float
            The cut cost of removing the edge between the two faces.
        """
        return self.edge_length(other) * min((self.angle_to(other)/np.pi)**10, 1)

    def _compute_area(self):
        """Compute the area of this face.

        Returns
        -------
        float
            The area of the face.
        """
        ab = self.vertex_coords[1] - self.vertex_coords[0]
        ac = self.vertex_coords[2] - self.vertex_coords[0]
        return 0.5 * np.linalg.norm(np.cross(ab, ac))

    def _compute_normal(self):
        """Compute the normal vector for this face.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The normalized 3D vector that is normal to this face.
        """
        ab = self.vertex_coords[1] - self.vertex_coords[0]
        ac = self.vertex_coords[2] - self.vertex_coords[0]
        return np.cross(ab, ac) / np.linalg.norm(np.cross(ab, ac))

    def _compute_edges(self):
        """Return a list of the edges in this face.

        Returns
        -------
        :obj:`list` of :obj:`tuple` of int
            A list of three ordered tuples
            that contain the indices of vertices that form the endpoints of each
            edge of the face. The first entry in each tuple is always the smaller of
            the two indices in the tuple.
        """
        edges = []
        n =  len(self.vertex_inds)
        for i in range(n):
            edges.append(FaceNode._ordered_tuple(self.vertex_inds[i],
                                                 self.vertex_inds[(i + 1) % n]))
        return edges

    @staticmethod
    def _ordered_tuple(v1, v2):
        """Return a tuple containing v1 and v2 in ascending order.

        Returns
        -------
        :obj:`tuple`
            A tuple containing v1 and v2 in ascending order.
        """
        if v1 < v2:
            return (v1, v2)
        return (v2, v1)

class SegmentNode(object):
    """A single segment in a 3D mesh. This class is used for the nodes in a
    SegmentGraph.
    """

    def __init__(self, segment_indices, face_indices, cut_cost, area):
        """Create a new SegmentNode.

        Parameters
        ----------
        segment_indices : :obj:`set` of int or a single int
            The integer ID's for the original segments were combined to form
            this new segment.

        face_indices : :obj:`set` of int
            A set containing the indices of the faces of the original 3D mesh
            that are contained in this segment.

        cut_cost : float
            The un-normalized cut cost of this segment.

        area : float
            The area of this segment.
        """
        if type(segment_indices) == int:
            segment_indices = [segment_indices]
        self._segment_indices = set(segment_indices)
        self._face_indices = set(face_indices)
        self._cut_cost = cut_cost
        self._area = area

    @property
    def segment_indices(self):
        """:obj:`set` of int : The indices of the original segments that were combined
        to form this one.
        """
        return self._segment_indices

    @segment_indices.setter
    def segment_indices(self, segment_indices):
        if type(segment_indices) == int:
            segment_indices = [segment_indices]
        self._segment_indices = set(segment_indices)

    @property
    def face_indices(self):
        """:obj:`set` of int : The indices of the faces of the original 3D mesh that
        are contained in this segment.
        """
        return self._face_indices

    @property
    def area(self):
        """float : The area of the segment.
        """
        return self._area

    @property
    def cut_cost(self):
        """float : The un-normalized cut cost of the segment.
        """
        return self._cut_cost

    @property
    def ncut_cost(self):
        """float : The area-normalized cut cost of the segment.
        """
        return self._cut_cost / self._area

    def merge(self, other, shared_cut_cost):
        """Create a new SegmentNode by combining two adjacent segments.

        Parameters
        ----------
        other : :obj:`SegmentNode`
            An adjacent segment to combine with this one.

        Returns
        -------
        :obj:`SegmentNode`
            The new segment created by merging the existing segments.
        """
        segment_indices = self.segment_indices | other.segment_indices
        face_indices = self.face_indices | other.face_indices
        cut_cost = self.cut_cost + other.cut_cost - 2 * shared_cut_cost
        area = self.area + other.area
        return SegmentNode(segment_indices, face_indices, cut_cost, area)

class FaceGraph(object):
    """An edge-weighted graph between the faces of a mesh.
    """

    def __init__(self, mesh):
        """Create a face graph from a mesh.

        Parameters
        ----------
        :obj:`meshpy.Mesh3D`
            A triangular mesh to create the graph with.
        """
        fg = nx.Graph()
        edgemap = {}

        # Create nodes in map
        for i, tri in enumerate(mesh.triangles):
            # Create a face and add it to the graph.
            f = FaceNode(i, tri, mesh.vertices)
            fg.add_node(f)

            # Add the face's edges to the edge map
            for edge in f.edges:
                if edge in edgemap:
                    edgemap[edge].append(f)
                else:
                    edgemap[edge] = [f]

        # Create edges in face and segment map
        for f in fg.nodes_iter():
            for edge in f.edges:
                adj_list = edgemap[edge]
                for other in adj_list:
                    if other != f and f.index < other.index:
                        fg.add_edge(f, other, cut_cost = f.cut_cost(other))

        self._graph = fg

    @property
    def graph(self):
        """:obj:`networkx.Graph` : The undirected graph between faces of the
        mesh. Edges are simply cut costs.
        """
        return self._graph

    @property
    def n_faces(self):
        """int : The number of faces in the graph.
        """
        return self.graph.number_of_nodes()

    @property
    def face_nodes(self):
        return self.graph.nodes()

    def edge_cut_cost(self, face1, face2):
        """float : The cost of cutting the link between face1 and face2.

        Parameters
        ----------
        face1 : :obj:`FaceNode`
            The first face.

        face2 : :obj:`FaceNode`
            The second face, adjacent to the first.

        Returns
        -------
        float
            The cost of cutting the edge between these faces.
        """
        return self.graph[face1][face2]['cut_cost']

    def neighbors(self, face):
        """Return a list of neighbors to the given face.

        Parameters
        ----------
        face : :obj:`FaceNode`
            The face to fetch neighbors of.

        Returns
        -------
        list of :obj:`FaceNode`
            A list of the faces adjacent to the given one.
        """
        return self.graph[face]

class SegmentGraph(object):
    """An edge-weighted graph between the adjacent segments of a mesh.
    """

    def __init__(self, face_graph, face_to_segment=None, existing_graph=None):
        """Create a segment graph from a mesh.

        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            A 3D triangular mesh to build a segment graph on.

        face_graph : :obj:`FaceGraph`
            A FaceGraph on which this SegmentGraph will be built.

        face_to_segment : :obj:`list` of int or ints, optional
            A list with one entry per face. The entry specifies
            the id of the segment that the corresponding mesh face
            is in. If omitted, each face will start in its own segment
            with ID numbers ascending from zero.

        existing_graph : :obj:`SegmentGraph`
            If included, an existing SegmentGraph whose structure will be
            copied to quickly build this new SegmentGraph.
        """
        self._face_graph = face_graph

        # Create seg graph from scratch or copy it from existing graph
        if existing_graph is None:
            self._seg_graph = self._create_seg_graph(face_to_segment)
        else:
            self._seg_graph = existing_graph.seg_graph.copy()

        # Create edge heap
        self._edge_heap = []
        for edge in self._seg_graph.edges_iter():
            s1, s2 = edge
            heapq.heappush(self._edge_heap, (-self._savings_on_merge(s1,s2), edge))

    @property
    def seg_graph(self):
        """:obj:`networkx.Graph` : The segment graph created for the mesh.
        """
        return self._seg_graph

    @property
    def segment_nodes(self):
        """:obj:`list` of :obj:`SegmentNode` : The segments in the graph.
        """
        return self.seg_graph.nodes()

    @property
    def n_segments(self):
        """int : The number of segments.
        """
        return self.seg_graph.number_of_nodes()

    def neighbors(self, seg):
        """Return a list of neighbors to the given segment.

        Parameters
        ----------
        segment : :obj:`SegmentNode`
            The segment to fetch neighbors of.

        Returns
        -------
        list of :obj:`SegmentNode`
            A list of the segments adjacent to the given one.
        """
        return self.seg_graph[seg]

    def merge_segments(self, s1, s2):
        """Merge two adjacent segments.

        Parameters
        ----------
        s1 : :obj:`SegmentNode`
            The first segment to merge.

        s2 : :obj:`SegmentNode`
            The second segment to merge.

        Raises
        ------
        ValueError
            If the segments are not adjacent.
        """
        if s1 not in self.neighbors(s2):
            raise ValueError('Merged segments must be adjacent.')

        shared_cost = self.seg_graph[s1][s2]['cut_cost']
        new_seg = s1.merge(s2, shared_cost)
        new_adj = list(set(self.neighbors(s1)) | set(self.neighbors(s2)))
        new_adj.remove(s1)
        new_adj.remove(s2)

        self.seg_graph.add_node(new_seg)
        for adj in new_adj:
            cut_cost = 0.0
            if adj in self.neighbors(s1):
                cut_cost += self.seg_graph[s1][adj]['cut_cost']
            if adj in self.neighbors(s2):
                cut_cost += self.seg_graph[s2][adj]['cut_cost']
            self.seg_graph.add_edge(new_seg, adj, cut_cost=cut_cost)
            savings = self._savings_on_merge(new_seg, adj)
            heapq.heappush(self._edge_heap, (-savings, (new_seg, adj)))

        self.seg_graph.remove_node(s1)
        self.seg_graph.remove_node(s2)

    def reindex_segment_nodes(self):
        """Re-index all segment nodes to start at 0 and increase to n-1,
        throwing away prior information about aggregated indices.
        """
        for i, seg_node in enumerate(self.seg_graph.nodes()):
            seg_node.segment_indices = i

    def cut_to_k_segments(self, k, deterministic=True):
        """Segment the mesh into k parts using hierarchical clustering.

        Parameters
        ----------
        k : int
            The number of segments in the desired final mesh.

        deterministic : bool
            If True, the best edge will be chosen every time.
            Otherwise, the edges are sorted and chosen stochastically, with
            higher probability placed on good edges.
        """
        while self.seg_graph.number_of_nodes() > k:
            if deterministic:
                edge = self._deterministic_min_edge()
            else:
                edge = self._stochastic_min_edge()
            self.merge_segments(edge[0], edge[1])

    def copy(self):
        """Return a copy of the segment graph, using the same face graph as
        before. This will throw away the segment history for each segment and
        only keep the smallest index for each.

        Returns
        -------
        :obj:`SegmentGraph`
            A copy of the existing segment graph.
        """
        return SegmentGraph(self._face_graph, existing_graph=self)

    def _savings_on_merge(self, s1, s2):
        """Compute the decrease in total cut cost if two segments are merged.

        Parameters
        ----------
        s1 : :obj:`SegmentNode`
            The first segment.

        s2 : :obj:`SegmentNode`
            The second segment.

        Returns
        -------
        float
            The (positive) decrease in total area-normalized cut cost for
            the current segmentation if the two segments were to be merged.
        """
        area = s1.area + s2.area
        original_cost = s1.ncut_cost + s2.ncut_cost
        cut_cost = self.seg_graph[s1][s2]['cut_cost']
        new_cost = (s1.cut_cost + s2.cut_cost - 2 * cut_cost) / area
        savings = original_cost - new_cost
        return savings

    def _deterministic_min_edge(self):
        """Return the minimum edge to relax on the graph.
        This runs in constant amortized time.

        Returns
        -------
        :obj:`tuple` of :obj:`SegmentNode`
            A tuple containing the two segments on the edge to relax.
        """
        while True:
            edge = heapq.heappop(self._edge_heap)[1]
            if edge[0] in self.seg_graph and edge[1] in self.seg_graph:
                return edge

    def _stochastic_min_edge(self):
        """Return the minimum edge to relax on the graph stochastically.
        This runs in linear time proportional to the number of segments.

        TODO: Make this faster.

        Returns
        -------
        :obj:`tuple` of :obj:`SegmentNode`
            A tuple containing the two segments on the edge to relax.
        """
        edge_cost_map = {}
        min_cost_diff = float("inf")
        max_cost_diff = 0.0
        for edge in self.seg_graph.edges_iter():
            s1, s2 = edge
            cost_diff = self._savings_on_merge(s1, s2)
            edge_cost_map[edge] = cost_diff
            if cost_diff < min_cost_diff:
                min_cost_diff = cost_diff
            if cost_diff > max_cost_diff:
                max_cost_diff = cost_diff

        edges = []
        probs = []
        prob_sum = 0.0
        for edge, cost in edge_cost_map.iteritems():
            prob = ((cost - min_cost_diff) / (max_cost_diff - min_cost_diff))**50
            edges.append(edge)
            probs.append(prob)
            prob_sum += prob
        probs = probs / prob_sum
        index = np.random.choice(range(len(edges)), 1, p=probs)[0]
        return edges[index]

    def _create_seg_graph(self, face_to_segment):
        """Create a segment graph from a face to segment map.

        Parameters
        ----------
        face_to_segment : :obj:`list` of int or ints, optional
            A list with one entry per face. The entry specifies
            the id of the segment that the corresponding mesh face
            is in. If omitted, each face will start in its own segment
            with ID numbers ascending from zero.

        Returns
        -------
        :obj:`networkx.graph`
            A graph of SegmentNode types.
        """
        fg = self._face_graph
        sg = nx.Graph()
        if face_to_segment is None:
            face_to_segment = range(self._face_graph.n_faces)

        seg_ids = set(face_to_segment)

        # A map from segment ID numbers to lists of faces in each segment
        segment_to_faces = {index : [] for index in seg_ids}

        # A map from segment ID numbers to sets of neighboring segment ID's
        neighbor_seg_ids = {index : set() for index in seg_ids}

        # Fill the two maps
        for face in fg.face_nodes:
            seg_id = face_to_segment[face.index]
            segment_to_faces[seg_id].append(face)
            for other_face in fg.neighbors(face):
                neighbor_seg_id = face_to_segment[other_face.index]
                if (neighbor_seg_id != seg_id):
                    neighbor_seg_ids[seg_id].add(neighbor_seg_id)

        # Create segments
        segs = {}
        for i, faces in segment_to_faces.iteritems():
            # Compute the area and cut cost of the segment
            area = 0.0
            cut_cost = 0.0
            for face in faces:
                area += face.area
                for other_face in fg.neighbors(face):
                    if other_face not in faces:
                        cut_cost += fg.edge_cut_cost(face, other_face)

            s = SegmentNode(i, [f.index for f in faces], cut_cost, area)
            sg.add_node(s)
            segs[i] = s

        # Use neighbor map to link segments
        for seg in sg.nodes():
            seg_id = min(seg.segment_indices)
            for neighbor_id in neighbor_seg_ids[seg_id]:
                neighbor_seg = segs[neighbor_id]
                if seg_id < neighbor_id:
                    cut_cost = 0.0
                    for face in segment_to_faces[seg_id]:
                        for neighbor_face in fg.neighbors(face):
                            if neighbor_face in segment_to_faces[neighbor_id]:
                                cut_cost += fg.edge_cut_cost(face, neighbor_face)
                    sg.add_edge(seg, neighbor_seg, cut_cost=cut_cost)
        return sg

class Segment(object):
    """A single surface segment for a mesh. Each segment is a combination of
    patches from a particular PatchSet.
    """

    def __init__(self, patch_set, patch_inds, ncut_cost):
        """Create a new Segment from a subset of a PatchSet.

        Parameters
        ----------
        patch_set : :obj:`PatchSet`
            A PatchSet. The segment is a subset of the
            surface patches in the PatchSet.

        patch_inds : :obj:`set` of int
            A set of indices of the patches in patch_set that are part of
            this segment.
        """
        self._patch_set = patch_set
        self._patches = frozenset([patch_set.patches[i] for i in patch_inds])
        self._patch_inds = frozenset(patch_inds)
        self._ncut_cost = ncut_cost

        self._area = 0.0
        for patch in self._patches:
            self._area += patch.area

        self._mesh = self._create_mesh()
        self._d2_descriptor = D2Descriptor(self.mesh, n_samples=1024*1)
        self._repetition_count = 1
        self._weight = 0.0

    @property
    def patches(self):
        """:obj:`frozenset` of Patch : The patches that are part of this
        segment.
        """
        return self._patches

    @property
    def patch_inds(self):
        """:obj:`frozenset` of int : The indices of the patches that are
        part of this segment.
        """
        return self._patch_inds

    @property
    def mesh(self):
        """:obj:`meshpy.Mesh3D` : A 3D mesh corresponding to this segment.
        """
        return self._mesh

    @property
    def area(self):
        """float : The surface area of this segment.
        """
        return self._area

    @property
    def ncut_cost(self):
        """float : The surface area of this segment.
        """
        return self._ncut_cost

    @property
    def d2_descriptor(self):
        """:obj:`D2Descriptor` : A D2 descriptor for the segment.
        """
        return self._d2_descriptor

    @property
    def repetition_count(self):
        """int : How many times this segment is seen in a set of random
        segmentations (set by the user).
        """
        return self._repetition_count

    @repetition_count.setter
    def repetition_count(self, rc):
        self._repetition_count = rc

    @property
    def weight(self):
        """float : A segmentation weight for this segment (set by the user).
        """
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    def intersects(self, other_seg):
        """Checks if this segment overlaps another one from the same PatchSet.

        Parameters
        ----------
        other_seg : :obj:`Segment`
            Another segment.

        Returns
        -------
        bool
            True if the segments overlap, False otherwise.
        """
        return bool(self.patches & other_seg.patches)

    def adjacent_to(self, other_seg):
        """Checks if this segment is adjacent to another one from the same PatchSet.

        Parameters
        ----------
        other_seg : :obj:`Segment`
            Another segment.

        Returns
        -------
        bool
            True if the segments are adjacent, False otherwise.
        """
        if not self.intersects(other_seg):
            for my_patch in self.patches:
                for other_patch in other_seg.patches:
                    if my_patch.adjacent_to(other_patch):
                        return True
        return False

    def distance_to(self, other_seg):
        """Computes the shape distance between two segments.

        Parameters
        ----------
        other_seg : :obj:`Segment`
            Another segment.

        Returns
        -------
        float
            The distance between the two segments, as computed by the segment's
            D2 shape descriptor.
        """
        return self.d2_descriptor.distance_to(other_seg.d2_descriptor)

    def show(self, color=(0.5, 0.5, 0.5)):
        """Render the 3D mesh for the segment using mayavi.

        Parameters
        ----------
        color : :obj:`tuple` of float
            RGB values in [0,1] for the color of the mesh.
        """
        Visualizer3D.mesh(self.mesh, style='surface', color=color, opacity=1.0)
        Visualizer3D.show()

    def _create_mesh(self):
        tri_inds = []
        for patch in self.patches:
            for tri_ind in patch.tri_inds:
                tri_inds.append(tri_ind)
        tris = [self._patch_set.mesh.triangles[i] for i in tri_inds]
        m = Mesh3D(self._patch_set.mesh.vertices, tris)
        m.remove_unreferenced_vertices()
        return m

    def __contains__(self, patch):
        return patch in self.patches

class Patch(object):
    """A subset of a mesh's surface, as part of a PatchSet. Patches are used
    to generate segments.
    """

    def __init__(self, patch_node, patch_set):
        """Create a new Patch.

        Parameters
        ----------
        patch_node : :obj:`SegmentNode`
            The SegmentNode representing this patch, retrieved from the
            PatchSet's SegmentGraph.

        patch_seg : :obj:`PatchSet`
            The PatchSet that this Patch belongs to.
        """
        self._patch_node = patch_node
        self._patch_set = patch_set
        self._index = min(patch_node.segment_indices)
        self._tri_inds = patch_node.face_indices
        self._area = patch_node.area
        self._mesh = None

    @property
    def index(self):
        """int : The integer index for this patch.
        """
        return self._index

    @property
    def tri_inds(self):
        """:obj:`list` of int : The triangle face indices for the faces in the
        PatchSet's mesh that are included in this Patch.
        """
        return self._tri_inds

    @property
    def area(self):
        """float : The surface area of this Patch.
        """
        return self._area

    @property
    def mesh(self):
        """:obj:`meshpy.Mesh3D` : A 3D mesh corresponding to this Patch.
        """
        if self._mesh is None:
            tris = [self._patch_set.mesh.triangles[i] for i in self.tri_inds]
            m = Mesh3D(self._patch_set.mesh.vertices, tris)
            m.remove_unreferenced_vertices()
            self._mesh = m
        return self._mesh

    def adjacent_to(self, other_patch):
        """Checks if this patch is adjacent to another one from the same PatchSet.

        Parameters
        ----------
        other_patch : :obj:`Patch`
            Another patch.

        Returns
        -------
        bool
            True if the patches are adjacent, False otherwise.
        """
        graph = self._patch_set._patch_graph
        return self._patch_node in graph.neighbors(other._patch_node)

class PatchSet(object):
    """A set of surface patches of a 3D mesh. PatchSet objects can generate
    random segments, which are subsets of their surface patches.
    """

    def __init__(self, mesh, n_patches):
        """Create a new PatchSet.

        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            The 3D mesh to make a PatchSet of.

        n_patches : int
            The desired number of patches in the patch set.
        """
        self._mesh = mesh
        self._n_patches = n_patches

        self._patch_graph = SegmentGraph(FaceGraph(mesh))
        self._patch_graph.cut_to_k_segments(self._n_patches)
        self._patch_graph.reindex_segment_nodes()

        self._patches = [Patch(node, self) for node in self._patch_graph.segment_nodes]
        self._patches.sort(key=lambda patch : patch.index)

    @property
    def patches(self):
        """:obj:`list` of Patch : A list of the Patches in the PatchSet.
        """
        return self._patches

    @property
    def mesh(self):
        """:obj:`meshpy.Mesh3D` : The original 3D mesh.
        """
        return self._mesh

    def show(self):
        """Visualize the PatchSet in 3D using mayavi.
        """
        for i, s in enumerate(self.patches):
            color = indexed_color(i)
            Visualizer3D.mesh(s.mesh, style='surface', color=color)
        Visualizer3D.show()

    def generate_random_segmentation(self, n_segments):
        """Generates a covering set of segments stochastically.

        Parameters
        ----------
        n_segments : int
            The number of segments in the final segmentation.

        Returns
        -------
        :obj:`list` of Segment
            A list of n_segments Segment objects.

        Raises
        ------
        ValueError
            If n_segments is invalid (<= 0 or > n_patches).
        """
        if n_segments > self._n_patches or n_segments <= 0:
            raise ValueError('Number of segments must be positive and >= number of patches.')

        seg_graph = self._patch_graph.copy()
        seg_graph.cut_to_k_segments(n_segments, False)


        segments = []
        for seg_node in seg_graph.segment_nodes:
            patch_inds = frozenset(seg_node.segment_indices)
            segments.append((patch_inds, seg_node.ncut_cost))

        return segments

class Shape(object):
    """A 3D shape wrapper for shape segmentation.
    """

    def __init__(self, mesh, n_patches):
        """Create a 3D shape and prepare it for segmentation.

        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            A 3D mesh.

        n_patches : int
            The number patches to start with.
        """
        self._mesh = mesh
        self._patch_set = PatchSet(self._mesh, n_patches)
        self._segmentations = []
        self._segments = {}

    @property
    def mesh(self):
        """:obj:`meshpy.Mesh3D` : The original 3D mesh.
        """
        return self._mesh

    @property
    def n_patches(self):
        """int : The number of patches in this Shape's PatchSet.
        """
        return len(self._patch_set.patches)

    @property
    def n_segments(self):
        """int : The number of segment candidates.
        """
        return len(self._segments)

    @property
    def patches(self):
        """:obj:`list` of :obj:`Patch` : This shape's patches.
        """
        return self._patch_set.patches

    @property
    def segments(self):
        """:obj:`list` of :obj:`Segment` : This shape's segment candidates.
        """
        return self._segments.values()

    def generate_random_segmentations(self, n_segmentations, n_segments):
        """Create a number of random segmentations and append them
        to the set of candidates for this shape.

        Parameters
        ----------
        n_segmentations : int
            The number of random segmentations to perform.

        n_segments : int
            The target number of segments for each of the random segmentations.
        """
        for i in range(n_segmentations):
            segmentation = self._patch_set.generate_random_segmentation(n_segments)
            cleaned_segmentation = []
            for patch_inds, ncut_cost in segmentation:
                if patch_inds not in self._segments:
                    seg = Segment(self._patch_set, patch_inds, ncut_cost)
                    self._segments[patch_inds] = seg
                    cleaned_segmentation.append(seg)
                else:
                    existing_segment = self._segments[patch_inds]
                    existing_segment.repetition_count += 1
                    cleaned_segmentation.append(existing_segment)
            self._segmentations.append(cleaned_segmentation)

    def prune_to(self, target_n_segs):
        final_segs = set(sorted(self.segments, key=lambda x : x.weight)[-target_n_segs:])

        min_segmentation = None
        max_min_segment_score = 0.0
        # Find segmentation with highest minimum segment score
        for segmentation in self._segmentations:
            min_segment_score = min([seg.weight for seg in segmentation])
            if min_segment_score > max_min_segment_score:
                min_segmentation = segmentation
                max_min_segment_score = min_segment_score

        for seg in min_segmentation:
            if seg not in final_segs:
                final_segs.add(seg)

        self._segmentations = []
        self._segments = {}
        for seg in final_segs:
            self._segments[seg.patch_inds] = seg

