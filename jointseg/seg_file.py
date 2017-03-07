"""
File for loading and saving mesh segmentations.
Author: Matthew Matl
"""
import os

from segmentation import MeshSegment, MeshSegmentation

class SegFile:
    """A mesh segmentation file reader and writer.
    """

    def __init__(self, filepath):
        """Construct and initialize a .seg file reader and writer.

        Parameters
        ----------
        filepath : :obj:`str`
            The full path to the desired .seg file

        Raises
        ------
        ValueError
            If the file extension is not .seg.
        """
        self._filepath = filepath
        file_root, file_ext = os.path.splitext(self._filepath)
        if file_ext != '.seg':
            raise ValueError('Extension %s invalid for SEGs' %(file_ext))

    @property
    def filepath(self):
        """:obj:`str` : The full path to the .seg file associated with this reader/writer.
        """
        return self._filepath

    def read(self, mesh):
        """Reads in the .seg file and returns a Segmentation using the
        associated mesh.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The 3D mesh associated with this segmentation.

        Returns
        -------
        :obj:`MeshSegmentation`
            A segmentation created from the given mesh and the .seg file.
        """
        f = open(self._filepath, 'r')

        face_to_segment = []
        for line in f:
            vals = line.split()
            if len(vals) > 0:
                val = int(vals[0])
                face_to_segment.append(val)
        f.close()

        n_segs = max(face_to_segment) + 1

        seg_tri_inds = [[] for i in range(n_segs)]

        for tri_ind, seg_ind in enumerate(face_to_segment):
            seg_tri_inds[seg_ind].append(tri_ind)

        segments = []
        for tri_inds in seg_tri_inds:
            segments.append(MeshSegment(tri_inds, mesh))

        return MeshSegmentation(mesh, segments)

    def write(self, segmentation):
        """Writes a Segmentation object out to a .seg file format.

        Parameters
        ----------
        segmentation : :obj:`MeshSegmentation`
            A segmentation of a 3D mesh.
        """
        seg_ids = [0 for i in range(len(segmentation.mesh.triangles))]
        for i, seg in enumerate(segmentation.segments):
            for tri_ind in seg.tri_inds:
                seg_ids[tri_ind] = i

        f = open(self._filepath, 'w')

        for seg_id in seg_ids:
            f.write('%d\n' %(seg_id))

        f.close()

