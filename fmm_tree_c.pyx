import numpy as np

DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

cdef class NodeData:
    def __init__(self, num_list, xpos, zpos, gvals, tpts, pval, dx, dz, xc, zc):
        self.num_list = num_list
        self.xpos = xpos
        self.zpos = zpos
        self.gvals = gvals
        self.tpts = tpts
        self.pval = pval
        self.dx = dx
        self.dz = dz
        self.children = 0

        self.center = np.array([xc, zc])
        self.center.shape = (1, 2)
        self.nodscndlst = []
        self.xcfs = np.empty(0, dtype=np.float64)
        self.kcursf = np.empty(0, dtype=np.float64)
        self.kvals = np.empty((1, pval+2), dtype=np.float64)

    cpdef void has_children(self):
        self.children = 1

    cpdef void set_kvals(self, kvals):
        self.kvals = kvals


cdef class TreeData:
    def __init__(self, xpos, zpos, gvals, pval, mx, ep, nvorts, ccnt, rcnt):
        self.xpos = xpos
        self.zpos = zpos
        self.gvals = gvals
        self.pval = pval
        self.mx = mx
        self.ep = ep
        self.nvorts = nvorts

        xmin = np.min(xpos)
        xmax = np.max(xpos)
        zmin = np.min(zpos)
        zmax = np.max(zpos)

        mlvl = np.int(np.floor(np.log(nvorts) / np.log(4)))
        xmin = xmin * (1. - np.sign(xmin) * .005)
        xmax = xmax * (1. + np.sign(xmax) * .005)
        zmin = zmin * (1. - np.sign(zmin) * .005)
        zmax = zmax * (1. + np.sign(zmax) * .005)

        self.xmin = xmin
        self.xmax = xmax
        self.zmin = zmin
        self.zmax = zmax
        self.mlvl = mlvl
        self.ccnt = ccnt
        self.rcnt = rcnt
        self.dx = (xmax - xmin) / ccnt
        self.dz = (zmax - zmin) / rcnt

    cdef double[:] xslice(self, long[:] inds, int npts):
        cdef Py_ssize_t kk
        cdef double[:] xsub = np.empty(npts, dtype=np.float64)
        cdef int cnt = 0
        for kk in xrange(npts):
            xsub[cnt] = self.xpos[inds[kk]]
            cnt += 1
        return xsub

    cdef double[:] zslice(self, long[:] inds, int npts):
        cdef Py_ssize_t kk
        cdef double[:] zsub = np.empty(npts, dtype=np.float64)
        cdef int cnt = 0
        for kk in xrange(npts):
            zsub[cnt] = self.zpos[inds[kk]]
            cnt += 1
        return zsub

    cdef double[:] gslice(self, long[:] inds, int npts):
        cdef Py_ssize_t kk
        cdef double[:] gsub = np.empty(npts, dtype=np.float64)
        cdef int cnt = 0
        for kk in xrange(npts):
            gsub[cnt] = self.gvals[inds[kk]]
            cnt += 1
        return gsub


class Node:
    def __init__(self, loc_dat):
        self.children = []
        self.my_dat = loc_dat

    def add_child(self, other):
        self.children.append(other)

    def get_child(self, ind):
        return self.children[ind]

    def get_dat(self):
        return self.my_dat

    def get_child_dat(self, ind):
        return self.children[ind].my_dat


class Tree:
    def __init__(self, loc_dat):
        self.nodes = []
        self.my_dat = loc_dat

    def add_node(self, node):
        self.nodes.append(node)

    def get_node(self, ind):
        return self.nodes[ind]

    def get_dat(self):
        return self.my_dat

    def get_node_dat(self, ind):
        return self.nodes[ind].my_dat