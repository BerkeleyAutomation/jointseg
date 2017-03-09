"""
File for processing random mesh segmentations.
Author: Matthew Matl
"""
import heapq
import math
import os

import numpy as np
import networkx as nx
import cvxopt as cvx
from sympy import Matrix

from visualization import Visualizer3D
from meshpy import Mesh3D

from d2_descriptor import D2Descriptor
from descriptor_file import DescriptorFile
import seg_file
from color_picker import indexed_color

# Disable output from CVX solvers
cvx.solvers.options['show_progress'] = False
cvx.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8

class FaceNode(object):
    """A single triangle in a 3D mesh. This class is used as the node element in a FaceGraph.
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

        angle = np.arccos(min(self.normal.dot(other.normal), 1.0))
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
        return self.edge_length(other) * min((self.angle_to(other)/np.pi)**10, 1.0)

    def copy(self):
        """Return an identical copy of this face node.

        Returns
        -------
        :obj:`FaceNode`
            An identical copy of this face node.
        """
        return FaceNode(self.index, self.vertex_inds, self.vertex_coords)

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
                        fg.add_edge(f, other, cut_cost = f.cut_cost(other), edge_length=f.edge_length(other))

        self._graph = fg
        self._mesh = mesh

    @property
    def graph(self):
        """:obj:`networkx.Graph` : The undirected graph between faces of the
        mesh. Edges are simply cut costs.
        """
        return self._graph

    @property
    def mesh(self):
        return self._mesh

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

    def edge_length(self, face1, face2):
        """float : The length of the edge between face1 and face2.

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
        return self.graph[face1][face2]['edge_length']

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

class SegmentNode(object):
    """A single segment in a 3D mesh. This class is used for the nodes in a
    SegmentGraph.
    """

    def __init__(self, segment_indices, face_nodes, cut_cost, area, perimeter):
        """Create a new SegmentNode.

        Parameters
        ----------
        segment_indices : :obj:`set` of int or a single int
            The integer ID's for the original segments were combined to form
            this new segment. This is used to track segments as they're merged
            together.

        face_nodes : :obj:`set` of :obj:`FaceNode`
            A set containing the FaceNode nodes for the faces of the original 3D mesh
            that are contained in this segment.

        cut_cost : float
            The un-normalized cut cost of this segment.

        area : float
            The area of this segment.

        perimeter : float
            The perimeter of this segment.
        """
        if type(segment_indices) == int:
            segment_indices = [segment_indices]
        self._segment_indices = set(segment_indices)
        self._face_nodes = set(face_nodes)
        self._face_indices = set([fn.index for fn in face_nodes])
        self._cut_cost = cut_cost
        self._area = area
        self._perimeter = perimeter

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
    def face_nodes(self):
        """:obj:`set` of :obj:`FaceNode` : The FaceNodes contained in this segment.
        """
        return self._face_nodes

    @face_nodes.setter
    def face_nodes(self, fnodes):
        self._face_indices = set([fn.index for fn in fnodes])
        self._face_nodes = set(fnodes)

    @property
    def area(self):
        """float : The area of the segment.
        """
        return self._area

    @area.setter
    def area(self, area):
        self._area = area

    @property
    def perimeter(self):
        """float : The perimeter of the segment.
        """
        return self._perimeter

    @perimeter.setter
    def perimeter(self, perim):
        self._perimeter = perim

    @property
    def cut_cost(self):
        """float : The un-normalized cut cost of the segment.
        """
        return self._cut_cost

    @cut_cost.setter
    def cut_cost(self, cut_cost):
        self._cut_cost = cut_cost

    @property
    def ncut_cost(self):
        """float : The area-normalized cut cost of the segment.
        """
        return self._cut_cost / self._area

    def merge(self, other, shared_cut_cost, shared_perim):
        """Create a new SegmentNode by combining two adjacent segments.

        Parameters
        ----------
        other : :obj:`SegmentNode`
            An adjacent segment to combine with this one.

        shared_cut_cost : float
            The cut cost of the edge between the two segments.

        shared_perim : float
            The perimeter of the edge between the two segments.

        Returns
        -------
        :obj:`SegmentNode`
            The new segment created by merging the existing segments.
        """
        segment_indices = self.segment_indices | other.segment_indices
        face_nodes = self.face_nodes | other.face_nodes
        cut_cost = self.cut_cost + other.cut_cost - 2 * shared_cut_cost
        perim = self.perimeter + other.perimeter - 2 * shared_perim
        area = self.area + other.area
        return SegmentNode(segment_indices, face_nodes, cut_cost, area, perim)

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
        self._edge_heap = self._build_edge_heap()

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
        return self.seg_graph[seg].keys()

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
        shared_perim = self.seg_graph[s1][s2]['shared_perim']
        new_seg = s1.merge(s2, shared_cost, shared_perim)
        new_adj = list(set(self.neighbors(s1)) | set(self.neighbors(s2)))
        new_adj.remove(s1)
        new_adj.remove(s2)

        self.seg_graph.add_node(new_seg)
        for adj in new_adj:
            cut_cost = 0.0
            shared_perim = 0.0
            if adj in self.neighbors(s1):
                cut_cost += self.seg_graph[s1][adj]['cut_cost']
                shared_perim += self.seg_graph[s1][adj]['shared_perim']
            if adj in self.neighbors(s2):
                cut_cost += self.seg_graph[s2][adj]['cut_cost']
                shared_perim += self.seg_graph[s2][adj]['shared_perim']
            self.seg_graph.add_edge(new_seg, adj, cut_cost=cut_cost, shared_perim=shared_perim)
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

    def refine_all_edges(self, fuzzy_dist=0.25):
        seg_to_id_map = {}
        for i, seg in enumerate(self.segment_nodes):
            seg_to_id_map[seg] = i
        for seg in self.segment_nodes:
            for nseg in self.neighbors(seg):
                if nseg in self.neighbors(seg) and seg_to_id_map[nseg] > seg_to_id_map[seg]:
                    self._refine_edge(seg, nseg, fuzzy_dist)

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
            prob = ((cost - min_cost_diff) / (max_cost_diff - min_cost_diff))**5
            edges.append(edge)
            probs.append(prob)
            prob_sum += prob
        probs = probs / prob_sum
        index = np.random.choice(range(len(edges)), 1, p=probs)[0]
        return edges[index]

    def _refine_edge(self, s1, s2, fuzzy_dist=0.25):
        """Refine the edge between two segments using fuzzy cuts.

        Parameters
        ----------
        s1 : :obj:`SegmentNode`
            The first segment.

        s2 : :obj:`SegmentNode`
            The second segment.
        """

        # Create graphs of faces within each segment and create sinks for each
        # between.
        fg1 = nx.Graph()
        fg2 = nx.Graph()

        seg_1_border = set()
        seg_2_border = set()

        # Add face nodes to graphs
        for fn in s1.face_nodes:
            fg1.add_node(fn)
        for fn in s2.face_nodes:
            fg2.add_node(fn)

        # VISUALIZE SEGMENTS
        #self._visualize_face_node_set(s1.face_nodes, (0.0, 1.0, 0.0))
        #self._visualize_face_node_set(s2.face_nodes, (0.0, 0.0, 1.0))

        # Add connections to segment 1's graph
        for fn in s1.face_nodes:
            for nfn in self._face_graph.neighbors(fn):
                if nfn in s1.face_nodes:
                    fg1.add_edge(fn, nfn, weight=self._face_graph.edge_cut_cost(fn, nfn))
                elif nfn in s2.face_nodes:
                    seg_1_border.add(fn)

        # Add connections to segment 2's graph
        for fn in s2.face_nodes:
            for nfn in self._face_graph.neighbors(fn):
                if nfn in s2.face_nodes:
                    fg2.add_edge(fn, nfn, weight=self._face_graph.edge_cut_cost(fn, nfn))
                elif nfn in s1.face_nodes:
                    seg_2_border.add(fn)

        # Add connections from border nodes to source nodes with zero cut cost.
        seg_1_source = object()
        seg_2_source = object()
        fg1.add_node(seg_1_source)
        for fn in seg_1_border:
            fg1.add_edge(fn, seg_1_source, weight=0.0)
        fg2.add_node(seg_2_source)
        for fn in seg_2_border:
            fg2.add_edge(fn, seg_2_source, weight=0.0)

        # Run Dijkstra's Algorithm from each source, using the cut cost as a
        # weight.
        dists1 = nx.single_source_dijkstra_path_length(fg1, seg_1_source, weight='weight')
        dists2 = nx.single_source_dijkstra_path_length(fg2, seg_2_source, weight='weight')

        # Find furthest distance in each segment.
        max_dist_1 = max(dists1.values())
        max_dist_2 = max(dists2.values())

        # Create a new graph for both segments together representing the fuzzy
        # region -- cutoff term is fuzzy_dist. Attach each to sinks.
        seg_1_sink = object()
        seg_2_sink = object()

        fuzzyg = nx.Graph()
        fuzzyg.add_node(seg_1_sink)
        fuzzyg.add_node(seg_2_sink)
        for fn, dist in dists1.iteritems():
            if dist < fuzzy_dist * max_dist_1 and fn != seg_1_source:
                fuzzyg.add_node(fn)
        for fn, dist in dists2.iteritems():
            if dist < fuzzy_dist * max_dist_2 and fn != seg_2_source:
                fuzzyg.add_node(fn)

        visnodes1 = []
        visnodes2 = []
        for fn in fuzzyg.nodes_iter():
            if fn != seg_1_sink and fn != seg_2_sink:
                if fn in s1.face_nodes:
                    visnodes1.append(fn)
                else:
                    visnodes2.append(fn)
                for nfn in self._face_graph.neighbors(fn):
                    if nfn in fuzzyg.nodes():
                        fuzzyg.add_edge(fn, nfn,
                                capacity=self._face_graph.edge_cut_cost(fn, nfn))#/self._face_graph.edge_length(fn, nfn))
                    elif nfn in s1.face_nodes:
                        fuzzyg.add_edge(fn, seg_1_sink, capacity=np.inf)
                    elif nfn in s2.face_nodes:
                        fuzzyg.add_edge(fn, seg_2_sink, capacity=np.inf)

        #self._visualize_face_node_set(visnodes1, (0.0, 1.0, 0.0), translation=1)
        #self._visualize_face_node_set(visnodes2, (0.0, 0.0, 1.0), translation=1)

        # Run a minimum cut algorithm on the fuzzy region
        cut_value, partition = nx.minimum_cut(fuzzyg, seg_1_sink, seg_2_sink)
        fuzzy_nodes_1, fuzzy_nodes_2 = partition[0], partition[1]

        # Re-organize the two segments based on the partition. Collect face
        # nodes for each segment and re-compute the cut_cost, area, and
        # perimeter for each.
        nodes1 = set()
        nodes2 = set()
        for fn in s1.face_nodes:
            if fn in fuzzy_nodes_2:
                nodes2.add(fn)
            else:
                nodes1.add(fn)
        for fn in s2.face_nodes:
            if fn in fuzzy_nodes_1:
                nodes1.add(fn)
            else:
                nodes2.add(fn)

        #self._visualize_face_node_set(nodes1, (0.0, 1.0, 0.0), translation=2)
        #self._visualize_face_node_set(nodes2, (0.0, 0.0, 1.0), translation=2)
        #Visualizer3D.show()

        s1.face_nodes = nodes1
        s2.face_nodes = nodes2

        neighboring_segs = set(self.neighbors(s1)) | set(self.neighbors(s2))

        # Remove all edges connecting to s1 and s2
        for nsn in self.neighbors(s1):
            self.seg_graph.remove_edge(s1, nsn)

        for nsn in self.neighbors(s2):
            self.seg_graph.remove_edge(s2, nsn)

        # Update all links to s1 and s2, and update those segment's areas,
        # perimeters, and cut costs.
        segs = [s1, s2]
        for sn in segs:
            area = 0.0
            perim = 0.0
            cut_cost = 0.0
            for fn in sn.face_nodes:
                area += fn.area
                for nfn in self._face_graph.neighbors(fn):
                    if nfn not in sn.face_nodes:
                        nsn = None
                        for osn in neighboring_segs:
                            if nfn in osn.face_nodes:
                                nsn = osn
                                break
                        if nsn not in self.neighbors(sn):
                            self.seg_graph.add_edge(sn, nsn, cut_cost=0.0, shared_perim=0.0)

                        multiplier = 1.0
                        if nsn in segs:
                            multiplier = 0.5
                        self.seg_graph[sn][nsn]['cut_cost'] += multiplier * self._face_graph.edge_cut_cost(fn, nfn)
                        self.seg_graph[sn][nsn]['shared_perim'] += multiplier * self._face_graph.edge_length(fn, nfn)
                        cut_cost += self._face_graph.edge_cut_cost(fn, nfn)
                        perim += self._face_graph.edge_length(fn, nfn)
            sn.area = area
            sn.perimeter = perim
            sn.cut_cost = cut_cost

        # Re-populate the pq
        self._edge_heap = self._build_edge_heap()

    def _build_edge_heap(self, use_raw_cost=False):
        """Rebuild the edge heap used for merging segments together.

        Parameters
        ----------
        use_raw_cost : bool
            If True, edges are stored by perimeter-weighted cut cost.
            If False, edges are stored by the reduction in area-normalized cut
            cost that removing them would result in.
        """
        edge_heap = []
        for edge in self.seg_graph.edges_iter():
            seg1, seg2 = edge
            cost = 0.0
            if use_raw_cost:
                cost = self.seg_graph[seg1][seg2]['cut_cost']/self.seg_graph[seg1][seg2]['shared_perim'] 
            else:
                cost = self._savings_on_merge(seg1, seg2)
            heapq.heappush(edge_heap, (-cost, edge))
        return edge_heap

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
                if neighbor_seg_id != seg_id:
                    neighbor_seg_ids[seg_id].add(neighbor_seg_id)

        # Create segments
        segs = {}
        for i, faces in segment_to_faces.iteritems():
            # Compute the area and cut cost of the segment
            area = 0.0
            cut_cost = 0.0
            perim = 0.0
            for face in faces:
                area += face.area
                for other_face in fg.neighbors(face):
                    if other_face not in faces:
                        cut_cost += fg.edge_cut_cost(face, other_face)
                        perim += fg.edge_length(face, other_face)

            s = SegmentNode(i, faces, cut_cost, area, perim)
            sg.add_node(s)
            segs[i] = s

        # Use neighbor map to link segments
        for seg in sg.nodes():
            seg_id = min(seg.segment_indices)
            for neighbor_id in neighbor_seg_ids[seg_id]:
                neighbor_seg = segs[neighbor_id]
                if seg_id < neighbor_id:
                    cut_cost = 0.0
                    shared_perim = 0.0
                    for face in segment_to_faces[seg_id]:
                        for neighbor_face in fg.neighbors(face):
                            if neighbor_face in segment_to_faces[neighbor_id]:
                                cut_cost += fg.edge_cut_cost(face, neighbor_face)
                                shared_perim += fg.edge_length(face, neighbor_face)
                    sg.add_edge(seg, neighbor_seg, cut_cost=cut_cost, shared_perim=shared_perim)
        return sg

class Segment(object):
    """A single surface segment for a mesh, which consists of a subset of its
    faces. This class is used for internal optimization routines.
    """

    def __init__(self, seg_node, orig_mesh):
        """Create a new Segment from a seg_node and an original mesh.

        Parameters
        ----------
        seg_node : :obj:`SegmentNode`
            A SegmentNode that is used to instantiate the Segment.

        orig_mesh : :obj:`meshpy.Mesh3D`
            The original mesh that this segment is a part of.
        """
        self._cut_cost = seg_node.cut_cost
        self._seg_inds = frozenset(seg_node.segment_indices)
        self._tri_inds = seg_node.face_indices
        self._area = seg_node.area
        self._perimeter = seg_node.perimeter
        self._weight = 0.0

        self._mesh = Segment._create_mesh(orig_mesh, seg_node.face_indices)
        self._d2_descriptor = D2Descriptor(self._mesh, n_samples=10000)

    @property
    def mesh(self):
        """:obj:`meshpy.Mesh3D` : A 3D mesh corresponding to this segment.
        """
        return self._mesh

    @property
    def seg_inds(self):
        """:obj:`frozenset` of int : The segment indices that are part of this
        segment
        """
        return self._seg_inds

    @property
    def tri_inds(self):
        """:obj:`list` of int : A list of the triangle indices of the original mesh
            that are present in this segment.
        """
        return self._tri_inds

    @property
    def area(self):
        """float : The surface area of this segment.
        """
        return self._area

    @property
    def perimeter(self):
        """float : The perimeter of this segment.
        """
        return self._perimeter

    @property
    def cut_cost(self):
        """float : The surface area of this segment.
        """
        return self._cut_cost

    @property
    def d2_descriptor(self):
        """:obj:`D2Descriptor` : A D2 descriptor for the segment.
        """
        return self._d2_descriptor

    @property
    def weight(self):
        """float : A segmentation weight for this segment (set by the user).
        """
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

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

    @staticmethod
    def _create_mesh(orig_mesh, face_indices):
        tris = [orig_mesh.triangles[i] for i in face_indices]
        m = Mesh3D(orig_mesh.vertices, tris)
        m.remove_unreferenced_vertices()
        return m

class SegmentCandidateSet(object):
    """A set of segment candidates for a given mesh.
    """

    def __init__(self, mesh, max_n_segs):
        """Create a set of segment candidates for a given mesh.

        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            A 3D mesh.

        max_n_segs : int
            The maximum number of segments that could appear in this shape.
        """
        self._mesh = mesh
        self._max_n_segs = max_n_segs

        # Create segments
        self.seg_graph = SegmentGraph(FaceGraph(mesh))

        index_sets_to_segs = {} # Maps index sets to segments
        seg_intersections = {} # Indexing [seg1][seg2] tells you if the two segments intersect

        n = max_n_segs

        # Create initial segments
        self.seg_graph.cut_to_k_segments(n)
        self.seg_graph.reindex_segment_nodes()
        self.seg_graph.refine_all_edges(0.4)

        while n >= 2:
            for seg_node in self.seg_graph.segment_nodes:
                seg_inds = frozenset(seg_node.segment_indices)

                # If we haven't seen this segment before, create it
                if seg_inds not in index_sets_to_segs:
                    seg = Segment(seg_node, self._mesh)

                    # Find any intersections
                    seg_intersections[seg] = {}
                    for other_seg_inds, other_seg in index_sets_to_segs.iteritems():
                        if bool(seg_inds & other_seg_inds):
                            seg_intersections[seg][other_seg] = True
                            seg_intersections[other_seg][seg] = True
                        else:
                            seg_intersections[seg][other_seg] = False
                            seg_intersections[other_seg][seg] = False

                    index_sets_to_segs[seg_inds] = seg
            n -= 1
            self.seg_graph.cut_to_k_segments(n)

        self._segments = seg_intersections.keys()
        self._seg_intersections = seg_intersections

    @property
    def mesh(self):
        """:obj:`meshpy.Mesh3D` : The original 3D mesh.
        """
        return self._mesh

    @property
    def n_seginds(self):
        """int : The maximum number of segments in a single segmentation.
        """
        return self._max_n_segs

    @property
    def n_segments(self):
        """int : The number of segment candidates.
        """
        return len(self._segments)

    @property
    def segments(self):
        """:obj:`list` of :obj:`Segment` : The segment candidates.
        """
        return self._segments

    def segs_intersect(self, seg1, seg2):
        """Do the two given segments intersect?

        Parameters
        ----------
        seg1 : :obj:`Segment`
            The first segment.

        seg2 : :obj:`Segment`
            The second segment.

        Returns
        -------
        bool
            True if the segments intersect, False otherwise.
        """
        return self._seg_intersections[seg1][seg2]

class MeshSegment(object):
    """A single surface segment for a mesh, which consists of a subset of its faces.
    """

    def __init__(self, tri_inds, orig_mesh, d2_descriptor=None):
        """Create a MeshSegment from a list of triangle indices and an original
        mesh.

        Parameters
        ----------
        tri_inds : :obj:`list` of int
            A list of the triangle indices of the original mesh that are present
            in this segment.

        orig_mesh : :obj:`meshpy.Mesh3D`
            The 3D mesh that this segment is taken out of.

        d2_descriptor : :obj:`D2Descriptor`
            The D2 shape descriptor for this segment, if it was already
            computed.
        """
        self._tri_inds = tri_inds
        self._mesh = MeshSegment._create_mesh(orig_mesh, tri_inds)
        self._d2_descriptor = d2_descriptor

    @property
    def mesh(self):
        """:obj:`meshpy.Mesh3D` : A 3D mesh corresponding to this segment.
        """
        return self._mesh

    @property
    def tri_inds(self):
        """:obj:`list` of int : A list of the triangle indices of the original mesh
            that are present in this segment.
        """
        return self._tri_inds

    @property
    def d2_descriptor(self):
        """:obj:`D2Descriptor` : A D2 descriptor for the segment.
        """
        return self._d2_descriptor

    @d2_descriptor.setter
    def d2_descriptor(self, d2_descriptor):
        self._d2_descriptor = d2_descriptor

    def compute_d2_descriptor(self, n_samples=10000):
        """Compute the D2 Descriptor for the segment.

        Parameters
        ----------
        n_samples : int
            The number of samples to use in the D2 descriptor.
        """
        self.d2_descriptor = D2Descriptor(self._mesh, n_samples=n_samples)

    def show(self, color=(0.5, 0.5, 0.5)):
        """Render the 3D mesh for the segment using mayavi.

        Parameters
        ----------
        color : :obj:`tuple` of float
            RGB values in [0,1] for the color of the mesh.
        """
        Visualizer3D.mesh(self.mesh, style='surface', color=color, opacity=1.0)
        Visualizer3D.show()

    @staticmethod
    def _create_mesh(orig_mesh, face_indices):
        tris = [orig_mesh.triangles[i] for i in face_indices]
        m = Mesh3D(orig_mesh.vertices, tris)
        m.remove_unreferenced_vertices()
        return m

class MeshSegmentation(object):
    """A segmentation of a 3D mesh.
    """

    def __init__(self, mesh, segments):
        """Create a MeshSegmentation.

        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            The 3D mesh that is segmented.

        segments : :obj:`list` of :obj:`MeshSegment`
            A list of MeshSegments that cover the original mesh.
        """
        self._mesh = mesh
        self._segments = segments

    @property
    def mesh(self):
        """:obj:`meshpy.Mesh3D` : The 3D mesh that this segment was created from.
        """
        return self._mesh

    @property
    def segments(self):
        """segments : :obj:`list` of :obj:`MeshSegment` :
        A list of MeshSegments that cover the original mesh.
        """
        return self._segments

    def show(self):
        """Display the MeshSegmentation with mayavi.
        """
        Visualizer3D.figure(size=(400, 400))
        for i, segment in enumerate(self.segments):
            Visualizer3D.mesh(segment.mesh, style='surface', color=indexed_color(i))
        Visualizer3D.show()

    def write(self, seg_filename, descriptor_dir=None):
        """Write the MeshSegmentation out to a file.

        Parameters
        ----------
        seg_filename : :obj:`str`
            The path of the .seg file, which will contain a segment id
            for each face.

        descriptor_dir : :obj:`str`
            If not None, a directory in which descriptor files for each segment
            can be cached.
        """
        seg_file.SegFile(seg_filename).write(self)
        if descriptor_dir is not None:
            for i, seg in enumerate(self.segments):
                filename = os.path.join(descriptor_dir, '{}.d2'.format(i))
                DescriptorFile(filename).write(seg.d2_descriptor)

    @staticmethod
    def load(mesh, seg_filename, descriptor_dir=None):
        """Read in a MeshSegmentation from a file.

        Parameters
        ----------
        seg_filename : :obj:`str`
            The path of the .seg file, which contains a segment id
            for each face.

        descriptor_dir : :obj:`str`
            If not None, a directory in which descriptor files for each segment
            have been cached. They are named with the approriate segment indices
            starting from zero (i.e. 0.d2, 1.d2, ...)

        Returns
        -------
        :obj:`MeshSegmentation`
            A completed mesh segmentation.
        """
        mesh_seg = seg_file.SegFile(seg_filename).read(mesh)
        if descriptor_dir is not None:
            for i, seg in enumerate(mesh_seg.segments):
                filename = os.path.join(descriptor_dir, '{}.d2'.format(i))
                d2_descriptor = DescriptorFile(filename).read(seg.mesh)
                seg.d2_descriptor = d2_descriptor
        else:
            for seg in mesh_seg.segments:
                seg.compute_d2_descriptor()
        return mesh_seg

class GroupShapeSegmenter(object):
    """A segmenter for a group of similar shapes.
    """

    def __init__(self, meshes, max_n_segs=10):
        """Create a GroupShapeSegmenter.

        Parameters
        ----------
        meshes : :obj:`list` of :obj:`meshpy.Mesh3D`
            A list of similar 3D meshes that should be jointly segmented.

        max_n_segs : int
            The maximal number of segments expected in any segmentation in this group.
        """

        self._meshes = meshes
        self._shapes = []

        # Create shapes
        for mesh in self._meshes:
            self._shapes.append(SegmentCandidateSet(mesh, max_n_segs))

        # Maps segments to a dict of shapes, with values being the closest
        # segment in that shape and the distance between the segments.
        seg_to_min_dists = {}

        # Initialize maps
        for shape in self._shapes:
            for seg in shape.segments:
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
                        if dist < seg_to_min_dists[seg1][shape2][1]:
                            seg_to_min_dists[seg1][shape2] = (seg2, dist)
                        if dist < seg_to_min_dists[seg2][shape1][1]:
                            seg_to_min_dists[seg2][shape1] = (seg1, dist)

        # Set of all computed distances and cut cost to perimeter ratios
        distances = []
        ratios = []

        # Compute sigmas for cut cost to perimeter ratios and 
        for seg in seg_to_min_dists:
            ratio = seg.cut_cost / seg.perimeter
            ratios.append(ratio)
            for shape, tup in seg_to_min_dists[seg].iteritems():
                dist = tup[1]
                if dist != 0.0:
                    distances.append(dist)
        sigdist = 2*np.median(distances)**2

        sigratio = (np.log(8.0)/(np.mean(ratios) - np.std(ratios))**2)

        # Compute weights for each segment
        for seg in seg_to_min_dists:
            seg.weight = 0.0

            self_cut_term = 0.0

            for shape, tup in seg_to_min_dists[seg].iteritems():
                other_seg, dist = tup
                cut_cost = other_seg.cut_cost
                perimeter = other_seg.perimeter
                ratio = cut_cost / perimeter

                dist_term = np.exp(-1*dist**2/sigdist)
                # TODO HERE
                cut_term = np.exp(-1*ratio**2*sigratio)
                weight = dist_term * cut_term

                # TODO HERE
                if other_seg == seg:
                    self_cut_term = cut_term
                seg.weight += weight

            if seg.cut_cost / seg.perimeter > np.mean(ratios) + np.std(ratios):
                seg.weight = 0.0
            seg.weight *= self_cut_term

        # Compute segmentations for each shape
        self._segmentations = self._compute_segmentations()

    @property
    def segmentations(self):
        """:obj:list of :obj:`MeshSegmentation` : A list of mesh segmentations.
        """
        return self._segmentations

    def _compute_segmentations(self):
        """Compute MeshSegmentation objects for each shape.

        Returns
        -------
        :obj:list of :obj:`MeshSegmentation`
            A list of mesh segmentations.
        """
        segmentations = []
        for shape in self._shapes:
            segments = self._run_ss_opt(shape)
            mesh_segments = [MeshSegment(s.tri_inds, shape.mesh, s.d2_descriptor) for s in segments]
            segmentations.append(MeshSegmentation(shape.mesh, mesh_segments))
        return segmentations

    def _run_ss_opt(self, shape):
        """Run single-shape optimization for selecting segments.

        Parameters
        ----------
        shape : 
        """
        final_segments = []

        #wseg = np.array([seg.weight * seg.area / shape.mesh.surface_area() for seg in shape.segments])
        wseg = np.array([seg.weight * np.sqrt(seg.area / shape.mesh.surface_area()) for seg in shape.segments])

        # Generate A and B matrices -- Ax = B
        #   These form the covering constraints for our generated segments.
        A = np.zeros((shape.n_seginds, shape.n_segments))
        for segind in range(shape.n_seginds):
            for i, segment in enumerate(shape.segments):
                if segind in segment.seg_inds:
                    A[segind][i] = 1.0
        A = GroupShapeSegmenter._li_matrix(A)
        B = np.ones((A.shape[0], 1))

        # Generate A and B matrices -- Gx <= H
        #   These constrain each segment indicator to lie between zero and one.
        G = np.vstack((-1.0*np.eye(shape.n_segments), np.eye(shape.n_segments)))
        H = np.vstack((np.zeros((shape.n_segments, 1)), np.ones((shape.n_segments, 1))))

        # Generate CVX matrices and solve the linear program
        C = cvx.matrix(-1.0 * wseg)
        A, B = cvx.matrix(A), cvx.matrix(B)
        G, H = cvx.matrix(G), cvx.matrix(H)
        sol = cvx.solvers.lp(C, G, H, A, B, solver='glpk')
        x = np.squeeze(np.array(sol['x']))

        # Round off segment indicators
        indicators = {seg : ind for seg, ind in zip(shape.segments, x)}
        while len(indicators.keys()) > 0:
            best_seg = max(indicators, key=indicators.get)
            final_segments.append(best_seg)
            del indicators[best_seg]

            for seg in indicators.keys():
                if shape.segs_intersect(best_seg, seg):
                    del indicators[seg]

        return final_segments

    @staticmethod
    def _li_matrix(A):
        """Compute a linearly-independent version of the rows of A.

        Paramters
        ---------
        A : :obj:`numpy.ndarray`
            A numpy matrix.

        Returns
        -------
        :obj:`numpy.ndarray`
            The matrix with its linearly dependent rows removed.
        """
        rref, rowinds = Matrix(A.T).rref()
        rows = []
        for ind in rowinds:
            rows.append(A[ind])
        return np.array(rows)

