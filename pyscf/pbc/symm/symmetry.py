#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
from numpy.linalg import inv
from pyscf import __config__
from pyscf.symm.Dmatrix import *
from functools import reduce

SYMPREC = getattr(__config__, 'pbc_symm_symmetry_symprec', 1e-6)
XYZ = np.eye(3)

def is_right_hand_screw(c):
    '''
    check if coordinate system is right hand screw
    '''
    x = c[0]
    y = c[1]
    z = c[2]

    val = np.dot(np.cross(x,y), z)   
    if val > 0:
        return True
    else:
        return False

def transform_rot_a_to_r(cell, ops):
    '''
    transform rotation operator from (a1,a2,a3) system to (x,y,z) system
    '''
    a = cell.lattice_vectors().T
    b = XYZ
    if ops.ndim == 2:
        return transform_rot(ops,a,b)
    else:
        ops_r = [transform_rot(op,a,b) for op in ops]
        return np.asarray(ops_r)

def transform_rot_b_to_r(cell, op):
    '''
    transform rotation operator from (b1,b2,b3) system to (x,y,z) system
    '''
    b = cell.reciprocal_vectors().T
    r = XYZ
    return transform_rot(op,b,r)

def transform_rot_b_to_a(cell, op):
    '''
    transform rotation operator from (b1,b2,b3) system to (a1,a2,a3) system
    '''
    a = cell.lattice_vectors().T
    b = cell.reciprocal_vectors().T
    return transform_rot(op,b,a)

def transform_rot_a_to_b(cell, op):
    '''
    transform rotation operator from (a1,a2,a3) system to (b1,b2,b3) system
    '''
    a = cell.lattice_vectors().T
    b = cell.reciprocal_vectors().T
    return transform_rot(op,a,b)

def transform_rot(op, a, b):
    '''
    transform rotation operator from (a1,a2,a3) system to (b1,b2,b3) system
    '''
    P = np.dot(inv(b),a)
    R = reduce(np.dot,(P, op, inv(P))).round(15)
    if(np.amax(np.absolute(R-R.round())) > SYMPREC):
        raise RuntimeError("rotation matrix is wrong!")
    return R.round().astype(int)

def get_Dmat(op,l):
    '''
    Get Wigner D matrix
    Arguments:
        op : (3,3) ndarray
            rotation operator in (x,y,z) system
        l : int
            angular momentum
    '''
    c1 = XYZ
    c2 = np.dot(op, c1.T).T
    right_hand = is_right_hand_screw(c2)
    alpha,beta,gamma = get_euler_angles(c1,c2)
    D = Dmatrix(l, alpha, beta, gamma, reorder_p=True)
    if not right_hand:
        if l == 1:
            D[:,0] *= -1.0
        elif l == 2:
            D[:,0] *= -1.0
            D[:,3] *= -1.0
        elif l == 3:
            D[:,1] *= -1.0
            D[:,4] *= -1.0
            D[:,6] *= -1.0
        elif l == 4:
            D[:,0] *= -1.0
            D[:,2] *= -1.0
            D[:,5] *= -1.0
            D[:,7] *= -1.0
        elif l > 4:
            raise NotImplementedError("l > 4 NYI")
    return D.round(15)

def get_Dmat_cart(op,l_max): #need verification
    pp = get_Dmat(op, 1)
    Ds = [np.ones((1,1))]
    for l in range(1, l_max+1):
        # All possible x,y,z combinations
        cidx = np.sort(lib.cartesian_prod([(0, 1, 2)] * l), axis=1)

        addr = 0
        affine = np.ones((1,1))
        for i in range(l):
            nd = affine.shape[0] * 3
            affine = np.einsum('ik,jl->ijkl', affine, pp).reshape(nd, nd)
            addr = addr * 3 + cidx[:,i]

        uniq_addr, rev_addr = np.unique(addr, return_inverse=True)
        ncart = (l + 1) * (l + 2) // 2
        assert ncart == uniq_addr.size
        trans = np.zeros((ncart,ncart))
        for i, k in enumerate(rev_addr):
            trans[k] += affine[i,uniq_addr]
        Ds.append(trans)
    return Ds

def make_Dmats(cell, ops, l_max=None):
    '''
    Computes < m | R | m' >
    '''
    if l_max is None:
        l_max = np.max(cell._bas[:,1])
    else:
        l_max = max(l_max, np.max(cell._bas[:,1]))
    nop = len(ops)
    Dmats = []

    for i in range(nop):
        op = transform_rot_a_to_r(cell, ops[i])
        if not cell.cart:
            Dmats.append([get_Dmat(op, l) for l in range(l_max+1)])
        else:
            Dmats.append(get_Dmat_cart(op, l_max))
    return Dmats, l_max

class SpaceGroup():
    '''
    Determines the space group of a lattice.
    Attributes:
        cell : pbc.cell.Cell class 
        symbol : string
        rotations : (n,3,3) ndarray 
        translations : (n,3) ndarray
    '''
    def __init__(self, cell, symprec=SYMPREC):
        self.cell = cell
        self.symprec = symprec
        self.rotations = np.reshape(np.eye(3,dtype=int), (-1,3,3))
        self.translations = np.zeros((1,3))
        self.symbol = None

    def get_space_group(self):
        try:
            from pyscf.pbc.symm.pyscf_spglib import cell_to_spgcell, get_spacegroup, get_symmetry
            cell = cell_to_spgcell(self.cell)
            self.symbol = get_spacegroup(cell, symprec=self.symprec)
            symmetry = get_symmetry(cell, symprec=self.symprec)
            self.rotations = symmetry['rotations']
            self.translations = symmetry['translations']
        except:
           raise NotImplementedError("use spglib to determine space group for now")

        return self

class Symmetry():
    '''
    contains space group symmetry info of a Cell object
    '''

    def __init__(self, cell, auxcell=None, point_group = True, symmorphic = True):

        self.cell = cell
        if self.cell is None: #no cell info
            self.space_group = None
            self.op_rot = np.eye(3,dtype =int).reshape(1,3,3)
            self.op_trans = np.zeros((1,3))
            self.Dmats = None #this may give errors
            return
        #if self.cell.cart == True:
        #    raise NotImplementedError("No symmetry support for cartesian basis yet")
        #from pyscf.pbc.tools.pyscf_ase import get_space_group
        #self.space_group = get_space_group(self.cell)
        self.space_group = SpaceGroup(cell).get_space_group()
        self.op_rot = self.space_group.rotations
        self.op_trans = self.space_group.translations
        self.symmorphic = symmorphic
        if self.symmorphic:
            self.op_rot = self.op_rot[np.where((np.absolute(self.op_trans)<1.e-10).all(1))]
            self.op_trans = np.zeros((len(self.op_rot),3))
        else:
            raise NotImplementedError('no sub-translational symmetry for now')

        if not point_group:
            self.op_rot = np.eye(3,dtype =int).reshape(1,3,3)
            self.op_trans = np.zeros((1,3))

        l_max = None
        if auxcell is not None:
            l_max = np.max(auxcell._bas[:,1])
        self.Dmats, self.l_max = make_Dmats(self.cell, self.op_rot, l_max)

def is_eye(op):

    return ((op - np.eye(3,dtype=int)) == 0).all()

def is_inversion(op):

    return ((op + np.eye(3,dtype=int)) == 0).all()

def get_phase(cell, op, coords_scaled, kpt_scaled):
    natm = cell.natm
    phase = np.zeros([natm], dtype = np.complex128)
    atm_map = np.arange(natm)
    for iatm in range(natm):
        r = coords_scaled[iatm]
        op_dot_r = np.dot(op, r)
        op_dot_r_0 = np.mod(np.mod(op_dot_r, 1), 1)
        diff = np.einsum('ki->k', abs(op_dot_r_0 - coords_scaled))
        atm_map[iatm] = np.where(diff < SYMPREC)[0]
        r_diff = op_dot_r_0 - op_dot_r
        #sanity check
        assert(np.linalg.norm(r_diff - r_diff.round()) < SYMPREC)
        phase[iatm] = np.exp(-1j * np.dot(kpt_scaled, r_diff) * 2.0 * np.pi)
    return atm_map, phase

def get_rotation_mat(kd, ibz_kpt_scaled, mo_coeff_or_dm, op_idx):
    cell = kd.cell
    sg_symm = kd.sg_symm
    kpt_scaled = np.dot(inv(kd.op_rot[op_idx]), ibz_kpt_scaled)
    op = transform_rot_b_to_a(cell, kd.op_rot[op_idx])
    inv_op = inv(op)
    coords = cell.get_scaled_positions()
    atm_map, phases = get_phase(cell, inv_op, coords, kpt_scaled)

    dim = mo_coeff_or_dm.shape[0]
    mat = np.zeros([dim, dim], dtype=np.complex128)
    aoslice = cell.aoslice_by_atom()
    for iatm in range(cell.natm):
        jatm = atm_map[iatm]
        if iatm != jatm:
            #sanity check
            nao_i = aoslice[iatm][3] - aoslice[iatm][2]
            nao_j = aoslice[jatm][3] - aoslice[jatm][2]
            assert(nao_i == nao_j)
            nshl_i = aoslice[iatm][1] - aoslice[iatm][0]
            nshl_j = aoslice[jatm][1] - aoslice[jatm][0]
            assert(nshl_i == nshl_j)
            for ishl in range(nshl_i):
                shl_i = ishl + aoslice[iatm][0]
                shl_j = ishl + aoslice[jatm][0]
                l_i = cell._bas[shl_i,1]
                l_j = cell._bas[shl_j,1]
                assert(l_i == l_j)
        phase = phases[iatm]
        ao_off_i = aoslice[iatm][2]
        ao_off_j = aoslice[jatm][2]
        shlid_0 = aoslice[iatm][0]
        shlid_1 = aoslice[iatm][1]
        for ishl in range(shlid_0, shlid_1):
            l = cell._bas[ishl,1]
            D = sg_symm.Dmats[op_idx][l] * phase
            if not cell.cart:
                nao = 2*l + 1
            else:
                nao = (l+1)*(l+2)//2
            nz = cell._bas[ishl,3]
            for j in range(nz):
                mat[ao_off_i:ao_off_i+nao, ao_off_j:ao_off_j+nao] = D
                ao_off_i += nao
                ao_off_j += nao
        assert(ao_off_i == aoslice[iatm][3])
        assert(ao_off_j == aoslice[jatm][3])
    return mat.T.conj()

def symmetrize_mo_coeff(kd, ibz_kpt_scaled, mo_coeff, op_idx):
    '''
    get MO coefficients for a symmetry related k point
    '''
    mat = get_rotation_mat(kd, ibz_kpt_scaled, mo_coeff, op_idx)
    res = np.dot(mat, mo_coeff)
    return res

def symmetrize_dm(kd, ibz_kpt_scaled, dm, op_idx):
    '''
    get density matrix for a symmetry related k point
    '''
    mat = get_rotation_mat(kd, ibz_kpt_scaled, dm, op_idx)
    res = reduce(np.dot, (mat, dm, mat.T.conj()))
    return res

def make_rot_loc(l_max, key):
    l = np.arange(l_max+1)
    if 'cart' in key:
        dims = ((l+1)*(l+2)//2)**2
    elif 'sph' in key:
        dims = (l*2+1)**2
    else:  # spinor
        raise NotImplementedError

    rot_loc = numpy.empty(len(dims)+1, dtype=np.int32)
    rot_loc[0] = 0
    dims.cumsum(dtype=np.int32, out=rot_loc[1:])
    return rot_loc
