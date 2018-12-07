cdef class NodeData:
    cdef int ndtpts, hschldrn, ndfpts
    cdef int[:] ndnodscndlst
    cdef int[:] num_list
    cdef double[:] ndxpos, ndzpos, ndgvals, ndcenter
    cdef double complex[:] ndkvals
    cdef double nddx, nddz
    cdef double[:,:] ndxcfs
    cdef double complex[:,:] ndkcursf

cdef class NodeList:
    cdef list nodes
    cdef int ind

cdef class Node(NodeList):
    cdef NodeData my_dat
    cdef NodeList children

cdef class Tree(NodeList):
    cdef double[:] tdglbxpos, tdglbzpos, tdglbgvals
    cdef int tdpval, tdnvorts, tdmlvl
    cdef double tdmx, tdep
    cdef NodeList children

    cpdef void set_glbdat(self, double[:] xpos, double[:] zpos, double[:] gvals, int pval, double mx, double ep, int nvorts)

    cdef double[:] xslice(self, int[:] inds, int npts)
    cdef double[:] zslice(self, int[:] inds, int npts)
    cdef double[:] gslice(self, int[:] inds, int npts)