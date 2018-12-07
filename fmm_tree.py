import numpy as np


class NodeData:
    def __init__(self):
        self._num_list = []
        self._xpos = []
        self._zpos = []
        self._gvals = []
        self._tpts = []
        self._pval = []
        self._dx = []
        self._dz = []
        self._hschldrn = False

        self._center = []
        self._nodscndlst = []
        self._xcfs = np.empty(0, dtype=np.float64)
        self._kcursf = np.empty(0, dtype=np.float64)
        self._kvals = np.zeros(0, dtype=np.float64)

    def vec_set(self, num_list, xloc, zloc, gloc, tpts, pval, dx, dz, xc, zc):
        self._num_list = num_list
        self._xpos = xloc
        self._zpos = zloc
        self._gvals = gloc
        self._tpts = tpts
        self._pval = pval
        self._dx = dx
        self._dz = dz
        self._center = np.array([xc, zc])

    @property
    def num_list(self):
        return self._num_list

    @num_list.setter
    def num_list(self, ilist):
        self._num_list = ilist

    @property
    def xpos(self):
        return self._xpos

    @xpos.setter
    def xpos(self, ilist):
        self._xpos = ilist

    @property
    def zpos(self):
        return self._zpos

    @zpos.setter
    def zpos(self, ilist):
        self._zpos = ilist

    @property
    def gvals(self):
        return self._gvals

    @gvals.setter
    def gvals(self, ilist):
        self._gvals = ilist

    @property
    def tpts(self):
        return self._tpts

    @tpts.setter
    def tpts(self, ipts):
        self._tpts = ipts

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, idx):
        self._dx = idx

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, idx):
        self._dz = idx

    @property
    def hschldrn(self):
        return self._hschldrn

    @hschldrn.setter
    def hschldrn(self, ival):
        self._hschldrn = ival

    @property
    def kvals(self):
        return self._kvals

    @kvals.setter
    def kvals(self, ivec):
        self._kvals = ivec

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, ivec):
        self._center = ivec

    @property
    def nodscndlst(self):
        return self._nodscndlst

    @nodscndlst.setter
    def nodscndlst(self, ilist):
        self._nodscndlst = ilist

    @property
    def xcfs(self):
        return self._xcfs

    @xcfs.setter
    def xcfs(self, imat):
        self._xcfs = imat

    @property
    def kcursf(self):
        return self._kcursf

    @kcursf.setter
    def kcursf(self, imat):
        self._kcursf = imat


class TreeData:
    def __init__(self):
        self._glb_inds = []
        self._xpos = []
        self._zpos = []
        self._gvals = []
        self._pval = 0
        self._mx = 0.
        self._ep = 0.
        self._nvorts = 0

        self._xmin = 0.
        self._xmax = 0.
        self._zmin = 0.
        self._zmax = 0.
        self._mlvl = 0
        self._ccnt = 0
        self._rcnt = 0
        self._dx = 0.
        self._dz = 0.

    def define_bounds(self):
        xmin = np.min(self.xpos)
        xmax = np.max(self.xpos)
        zmin = np.min(self.zpos)
        zmax = np.max(self.zpos)
        self._glb_inds = np.arange(self.nvorts)
        self._mlvl = np.floor(np.log(self.nvorts) / np.log(4))
        self._xmin = xmin * (1. - np.sign(xmin) * .005)
        self._xmax = xmax * (1. + np.sign(xmax) * .005)
        self._zmin = zmin * (1. - np.sign(zmin) * .005)
        self._zmax = zmax * (1. + np.sign(zmax) * .005)
        self._dx = (self.xmax - self.xmin) / float(self.ccnt)
        self._dz = (self.zmax - self.zmin) / float(self.rcnt)

    @property
    def glb_inds(self):
        return self._glb_inds

    @glb_inds.setter
    def glb_inds(self, ilist):
        self._glb_inds = ilist

    @property
    def xpos(self):
        return self._xpos

    @xpos.setter
    def xpos(self, ilist):
        self._xpos = ilist

    @property
    def zpos(self):
        return self._zpos

    @zpos.setter
    def zpos(self, ilist):
        self._zpos = ilist

    @property
    def gvals(self):
        return self._gvals

    @gvals.setter
    def gvals(self, ilist):
        self._gvals = ilist

    @property
    def pval(self):
        return self._pval

    @pval.setter
    def pval(self, ival):
        self._pval = ival

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, idx):
        self._dx = idx

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, idx):
        self._dz = idx

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, val):
        self._mx = val

    @property
    def ep(self):
        return self._ep

    @ep.setter
    def ep(self, val):
        self._ep = val

    @property
    def rcnt(self):
        return self._rcnt

    @rcnt.setter
    def rcnt(self, val):
        self._rcnt = val

    @property
    def ccnt(self):
        return self._ccnt

    @ccnt.setter
    def ccnt(self, val):
        self._ccnt = val

    @property
    def nvorts(self):
        return self._nvorts

    @nvorts.setter
    def nvorts(self, val):
        self._nvorts = val

    @property
    def mlvl(self):
        return self._mlvl

    @mlvl.setter
    def mlvl(self, ival):
        self._mlvl = ival

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, val):
        self._xmin = val

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, val):
        self._xmax = val

    @property
    def zmin(self):
        return self._zmin

    @zmin.setter
    def zmin(self, val):
        self._zmin = val

    @property
    def zmax(self):
        return self._zmax

    @zmax.setter
    def zmax(self, val):
        self._zmax = val


class NodeList:

    def __init__(self):
        self.nodes = []
        self.ind = 0

    def __iadd__(self, x):
        self.nodes.append(x)
        return self

    def __getitem__(self, ind):
        return self.nodes[ind]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return self

    def next(self):
        if self.ind >= len(self):
            raise StopIteration()
        ret = self.nodes[self.ind]
        self.ind += 1
        return ret


class Node(NodeData, NodeList):
    def __init__(self):
        self.children = NodeList()
        self._my_dat = NodeData()

    @property
    def my_dat(self):
        return self._my_dat

    @my_dat.setter
    def my_dat(self, loc_dat):
        self._my_dat = loc_dat

    @property
    def num_list(self):
        return self.my_dat.num_list

    @property
    def xpos(self):
        return self.my_dat.xpos

    @property
    def zpos(self):
        return self.my_dat.zpos

    @property
    def gvals(self):
        return self.my_dat.gvals

    @property
    def dx(self):
        return self.my_dat.dx

    @property
    def dz(self):
        return self.my_dat.dz

    @property
    def tpts(self):
        return self.my_dat.tpts

    @property
    def center(self):
        return self.my_dat.center

    @property
    def kvals(self):
        return self.my_dat.kvals

    @property
    def nodscndlst(self):
        return self.my_dat.nodscndlst

    @nodscndlst.setter
    def nodscndlst(self, ilist):
        self.my_dat.nodscndlst = ilist

    @property
    def kcursf(self):
        return self.my_dat.kcursf

    @kcursf.setter
    def kcursf(self, imat):
        self.my_dat.kcursf = imat

    @property
    def xcfs(self):
        return self.my_dat.xcfs

    @xcfs.setter
    def xcfs(self, imat):
        self.my_dat.xcfs = imat

    @property
    def parent(self):
        return self.my_dat.hschldrn

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return self.children.nodes[index]

    def __len__(self):
        return len(self.children.nodes)

    def __iadd__(self, x):
        self.children += x
        return self

    def next(self):
        if self.children.ind >= len(self):
            raise StopIteration()
        ret = self.children.nodes[self.children.ind]
        self.ind += 1
        return ret


class Tree(TreeData, NodeList):
    def __init__(self):
        self.children = NodeList()
        self._my_dat = TreeData()

    @property
    def my_dat(self):
        return self._my_dat

    @my_dat.setter
    def my_dat(self, idat):
        self._my_dat = idat

    @property
    def glb_inds(self):
        return self.my_dat.glb_inds

    @property
    def xpos(self):
        return self.my_dat.xpos

    @property
    def zpos(self):
        return self.my_dat.zpos

    @property
    def gvals(self):
        return self.my_dat.gvals

    @property
    def nvorts(self):
        return self.my_dat.nvorts

    @property
    def pval(self):
        return self.my_dat.pval

    @property
    def mlvl(self):
        return self.my_dat.mlvl

    @property
    def mx(self):
        return self.my_dat.mx

    @property
    def ep(self):
        return self.my_dat.ep

    @property
    def dx(self):
        return self.my_dat.dx

    @property
    def dz(self):
        return self.my_dat.dz

    @property
    def xmin(self):
        return self.my_dat.xmin

    @property
    def xmax(self):
        return self.my_dat.xmax

    @property
    def zmin(self):
        return self.my_dat.zmin

    @property
    def zmax(self):
        return self.my_dat.zmax

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return self.children.nodes[index]

    def __len__(self):
        return len(self.children.nodes)

    def __iadd__(self, x):
        self.children += x
        return self

    def next(self):
        if self.children.ind >= len(self):
            raise StopIteration()
        ret = self.children.nodes[self.children.ind]
        self.ind += 1
        return ret
