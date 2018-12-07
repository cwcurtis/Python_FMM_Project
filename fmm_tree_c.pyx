import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.intc

cdef class NodeData:
    def __cinit__(self, int[:] num_list, double[:] xpos, double[:] zpos, double[:] gvals, int tpts, double dx, double dz, double xc, double zc):
        self.num_list = num_list
        self.ndxpos = xpos
        self.ndzpos = zpos
        self.ndgvals = gvals
        self.ndtpts = tpts
        self.nddx = dx
        self.nddz = dz
        self.hschldrn = 0
        self.ndfpts = 0

        self.ndcenter = np.empty(2, dtype=np.float64)
        self.ndcenter[0] = xc
        self.ndcenter[1] = zc
        self.ndnodscndlst = np.empty(0, dtype=DTYPE)
        self.ndxcfs = np.empty(2, dtype=np.float64).reshape(1,2)
        self.ndkcursf = np.empty((1, 1), dtype=np.complex128)
        self.ndkvals = np.empty(0, dtype=np.complex128)

cdef class NodeList:

    def __init__(self):
        self.nodes = []
        self.ind = 0

    def __iadd__(self, Node x):
        self.nodes.append(x)
        return self

    def __getitem__(self,ind):
        return self.nodes[ind]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.ind >= len(self):
            raise StopIteration()
        ret = self.nodes[self.ind]
        self.ind += 1
        return ret

cdef class Node(NodeList):

    def __init__(self, NodeData loc_dat):
        self.children = NodeList()
        self.my_dat = loc_dat

    property node_dat:
        def __get__(self):
            return self.my_dat

    property xpos:
        def __get__(self):
            return self.my_dat.ndxpos

    property zpos:
        def __get__(self):
            return self.my_dat.ndzpos

    property gvals:
        def __get__(self):
            return self.my_dat.ndgvals

    property kvals:
        def __get__(self):
            return self.my_dat.ndkvals

        def __set__(self, ivec):
            self.my_dat.ndkvals = ivec

    property tpts:
        def __get__(self):
            return self.my_dat.ndtpts

    property dx:
        def __get__(self):
            return self.my_dat.nddx

    property dz:
        def __get__(self):
            return self.my_dat.nddz

    property myinds:
        def __get__(self):
            return self.my_dat.num_list

    property nodscndlst:
        def __set__(self, inlst):
            self.my_dat.ndnodscndlst = inlst

        def __get__(self):
            return self.my_dat.ndnodscndlst

    property fpts:
        def __set__(self, ipts):
            self.my_dat.ndfpts = ipts

        def __get__(self):
            return self.my_dat.ndfpts

    property kcursf:
        def __set__(self, cmat):
            self.my_dat.ndkcursf = cmat

        def __get__(self):
            return self.my_dat.ndkcursf

    property xcfs:
        def __set__(self, cmat):
            self.my_dat.ndxcfs = cmat

        def __get__(self):
            return self.my_dat.ndxcfs

    property parent:
        def __get__(self):
            return self.my_dat.hschldrn

        def __set__(self, ival):
            self.my_dat.hschldrn = ival

    property center:
        def __get__(self):
            return self.my_dat.ndcenter

    def __iter__(self):
        return self

    def __getitem__(self,index):
        return self.children.nodes[index]

    def __len__(self):
        return len(self.children.nodes)

    def __iadd__(self, Node x):
        self.children += x
        return self

    def __next__(self):
        if self.children.ind >= len(self):
            raise StopIteration()
        ret = self.children.nodes[self.children.ind]
        self.ind += 1
        return ret

cdef class Tree(NodeList):

    def __cinit__(self):
        self.children = NodeList()

    cpdef void set_glbdat(self, double[:] xpos, double[:] zpos, double[:] gvals, int pval, double mx, double ep, int nvorts):
        self.tdglbxpos = xpos
        self.tdglbzpos = zpos
        self.tdglbgvals = gvals
        self.tdpval = pval
        self.tdmx = mx
        self.tdep = ep
        self.tdnvorts = nvorts

    property pval:
        def __get__(self):
            return self.tdpval

    property nvorts:
        def __get__(self):
            return self.tdnvorts

    property mx:
        def __get__(self):
            return self.tdmx

    property ep:
        def __get__(self):
            return self.tdep

    property mlvl:
        def __get__(self):
            return self.tdmlvl

    property glbxpos:
        def __get__(self):
            return self.tdglbxpos

    property glbzpos:
        def __get__(self):
            return self.tdglbzpos

    def __iter__(self):
        return self

    def __getitem__(self,index):
        return self.children.nodes[index]

    def __len__(self):
        return len(self.children.nodes)

    def __iadd__(self, Node x):
        self.children += x
        return self

    def __next__(self):
        if self.children.ind >= len(self):
            raise StopIteration()
        ret = self.children.nodes[self.children.ind]
        self.ind += 1
        return ret

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] xslice(self, int[:] inds, int npts):
        cdef Py_ssize_t kk
        cdef double[:] xsub = np.empty(npts, dtype=np.float64)
        cdef int cnt = 0
        for kk in range(npts):
            xsub[cnt] = self.tdglbxpos[inds[kk]]
            cnt += 1
        return xsub

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] zslice(self, int[:] inds, int npts):
        cdef Py_ssize_t kk
        cdef double[:] zsub = np.empty(npts, dtype=np.float64)
        cdef int cnt = 0
        for kk in xrange(npts):
            zsub[cnt] = self.tdglbzpos[inds[kk]]
            cnt += 1
        return zsub

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] gslice(self, int[:] inds, int npts):
        cdef Py_ssize_t kk
        cdef double[:] gsub = np.empty(npts, dtype=np.float64)
        cdef int cnt = 0
        for kk in xrange(npts):
            gsub[cnt] = self.tdglbgvals[inds[kk]]
            cnt += 1
        return gsub