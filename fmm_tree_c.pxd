cdef class NodeData:
    cdef public double[:] xpos, zpos, gvals
    cdef public int tpts, children, pval
    cdef public double dx, dz, xc, zc
    cdef public object  num_list, center, nodscndlst, xcfs, kcursf, kvals
    cpdef void has_children(self)
    cpdef void set_kvals(self, kvals)

cdef class TreeData:
    cdef public double[:] xpos, zpos, gvals
    cdef public int pval, nvorts, ccnt, rcnt, mlvl
    cdef public double mx, xmin, xmax, zmin, zmax, dx, dz, ep
    cdef double[:] xslice(self, long[:] inds, int npts)
    cdef double[:] zslice(self, long[:] inds, int npts)
    cdef double[:] gslice(self, long[:] inds, int npts)