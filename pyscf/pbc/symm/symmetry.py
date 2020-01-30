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
# Authors: Xing Zhang
#

import numpy as np
from numpy.linalg import inv
from pyscf.symm.Dmatrix import *
from functools import reduce

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

def transform_rot_a_to_r(cell, op):
    '''
    transform rotation operator from (a1,a2,a3) system to (x,y,z) system
    '''
    a = cell.lattice_vectors().T
    b = XYZ
    return transform_rot(op,a,b)

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
    if(np.amax(np.absolute(R-R.round())) > 1e-6):
        raise RuntimeError("rotation matrix is wrong!")
    return R.astype(int)

def get_Dmat(op,l):
    '''
    Get Wigner D matrix
    Arguments:
        op : in (x,y,z) system
        l  : angular momentum
    '''
    c1 = XYZ
    c2 = np.dot(op, c1.T).T
    right_hand = is_right_hand_screw(c2)

    alpha,beta,gamma = get_euler_angles(c1,c2)
    D = Dmatrix(l, alpha, beta, gamma, reorder_p=True)
    if not right_hand:
        if l == 1:
            D[:,0] *= -1.0
        if l == 2:
            D[:,0] *= -1.0
            D[:,3] *= -1.0
        if l == 3:
            D[:,1] *= -1.0
            D[:,4] *= -1.0
            D[:,6] *= -1.0
        if l > 3:
            raise NotImplementedError("l > 3 NYI")
    return D.round(15)

def make_Dmats(cell, ops):
    '''
    Computes < m | R | m' >
    '''
    l_max = np.max(cell._bas[:,1])
    nop = len(ops)
    Dmats = []

    for i in range(nop):
        op = transform_rot_a_to_r(cell, ops[i])
        Dmats.append([])
        for l in range(l_max+1):
            Dmats[i].append(get_Dmat(op, l))
    return Dmats

def cell_to_spgcell(cell):
    a = cell.lattice_vectors()
    atm_pos = cell.get_scaled_positions()
    atm_num = []
    from pyscf.data import elements
    for symbol in cell.elements:
        atm_num.append(elements.NUC[symbol])
    spg_cell = (a,atm_pos,atm_num)
    return spg_cell

class SpaceGroup():
    '''
    Determines the space group of a lattice.
    Attributes:
        cell : pbc.cell.Cell class 
        symbol : string
        rotations : (n,3,3) ndarray 
        translations : (n,3) ndarray
    '''
    def __init__(self, cell, symprec=1e-6):
        self.cell = cell
        self.symprec = symprec
        self.rotations = np.reshape(np.eye(3,dtype=int), (-1,3,3))
        self.translations = np.zeros((1,3))
        self.symbol = None

    def get_space_group(self):
        try:
            import spglib
            cell = cell_to_spgcell(self.cell)
            self.symbol = spglib.get_spacegroup(cell, symprec=self.symprec)
            symmetry = spglib.get_symmetry(cell, symprec=self.symprec)
            self.rotations = symmetry['rotations']
            self.translations = symmetry['translations']
        except:
            raise NotImplementedError("use spglib to determine space group for now")

        return self

class Symmetry():
    '''
    contains space group symmetry info of a Cell object
    '''

    def __init__(self, cell, point_group = True, symmorphic = True):

        self.cell = cell
        if self.cell is None: #no cell info
            self.space_group = None
            self.op_rot = np.eye(3,dtype =int).reshape(1,3,3)
            self.op_trans = np.zeros((1,3))
            self.Dmats = None #this may give errors
            return
        if self.cell.cart == True:
            raise NotImplementedError("No symmetry support for cartesian basis yet")
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

        self.Dmats = make_Dmats(self.cell, self.op_rot)

def is_eye(op):

    return ((op - np.eye(3,dtype=int)) == 0).all()

def is_inversion(op):

    return ((op + np.eye(3,dtype=int)) == 0).all()

def _get_phase(cell, ishell, op, coords_scaled, kpt_scaled):
    iatm = cell._bas[ishell,0]
    r = coords_scaled[iatm]
    r_diff = r - np.dot(op, r)
    phase = np.exp(-1j * np.dot(kpt_scaled, r_diff) * 2.0 * np.pi)
    return phase

def symmetrize_mo_coeff(kd, ibz_kpt_scaled, mo_coeff, op_idx):
    '''
    get MO coefficients for a symmetry related k point
    '''
    cell = kd.cell
    sg_symm = kd.sg_symm
    kpt_scaled = np.dot(inv(kd.op_rot[op_idx]), ibz_kpt_scaled)
    op = transform_rot_b_to_a(cell, kd.op_rot[op_idx])
    inv_op = inv(op)
    coords = cell.get_scaled_positions()

    res = np.empty_like(mo_coeff)
    nshell = cell._bas.shape[0]
    ioff = 0
    for i in range(nshell):
        phase = _get_phase(cell, i, inv_op, coords, kpt_scaled)

        l = cell._bas[i,1]
        nao = 2*l + 1
        D = sg_symm.Dmats[op_idx][l] * phase
        nz = cell._bas[i,3]
        for j in range(nz):
            res[ioff:ioff+nao,:] = np.dot(D.T.conj(), mo_coeff[ioff:ioff+nao,:])
            ioff += nao
    return res

def symmetrize_dm(kd, ibz_kpt_scaled, dm, op_idx):
    '''
    get density matrix for a symmetry related k point
    '''
    cell = kd.cell
    sg_symm = kd.sg_symm
    kpt_scaled = np.dot(inv(kd.op_rot[op_idx]), ibz_kpt_scaled)
    op = transform_rot_b_to_a(cell, kd.op_rot[op_idx])
    inv_op = inv(op)
    coords = cell.get_scaled_positions()

    res = np.zeros_like(dm)
    nshell = cell._bas.shape[0]
    ioff = 0
    for i in range(nshell):
        phase_i = _get_phase(cell, i, inv_op, coords, kpt_scaled)

        l_i = cell._bas[i,1]
        nao_i = 2*l_i + 1
        nz_i = cell._bas[i,3]
        Di = sg_symm.Dmats[op_idx][l_i] * phase_i
        #Di = phase_i*Di.astype(np.complex128)
        for iz in range(nz_i):
            joff = 0
            for j in range(nshell):
                phase_j = _get_phase(cell, j, inv_op, coords, kpt_scaled)

                l_j = cell._bas[j,1]
                nao_j = 2*l_j + 1
                nz_j = cell._bas[j,3]
                Dj = sg_symm.Dmats[op_idx][l_j] * phase_j
                #Dj = (phase_j*Dj.astype(np.complex128)).conj()
                for jz in range(nz_j):
                    dm_k = reduce(np.dot,(Di.T.conj(), dm[ioff:ioff+nao_i,joff:joff+nao_j], Dj))
                    if res.dtype == np.double:
                        dm_k = dm_k.real
                    res[ioff:ioff+nao_i,joff:joff+nao_j] = dm_k
                    joff += nao_j
            ioff += nao_i
    return res
