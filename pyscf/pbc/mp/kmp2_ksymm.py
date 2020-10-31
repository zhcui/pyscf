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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import time
import numpy as np
from scipy.linalg import block_diag
from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.mp import kmp2
from pyscf.pbc.mp.kmp2 import WITH_T2

def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    t0 = (time.clock(), time.time())
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    kd = mp.kpts
    nbzk = kd.nkpts

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    emp2 = 0.
    oovv_ij = np.zeros((nbzk,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")

    kijab, weight, _ = kd.make_k4_ibz(sym='s2')
    if with_t2:
        #FIXME how many t2 do we need to store?
        t2 = np.zeros((len(kijab)*2, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    _, igroup = np.unique(kijab[:,:2], axis=0, return_index=True)
    igroup = list(igroup) + [len(kijab)]

    nao2mo = 0
    icount = 0
    for i in range(len(igroup)-1):
        istart = igroup[i]
        iend = igroup[i+1]
        kab = []
        for j in range(istart, iend):
            a, b = kijab[j][2:]
            kab.append([a, b])
            kab.append([b, a])
        kab = np.unique(np.asarray(kab), axis=0)

        ki_bz = kijab[istart][0]
        kj_bz = kijab[istart][1]
        kpts_i = kd.kpts[ki_bz]
        kpts_j = kd.kpts[kj_bz]
        ki = kd.bz2ibz[ki_bz]
        kj = kd.bz2ibz[kj_bz]
        orbo_i = kd.transform_single_mo_coeff(mo_coeff, ki_bz)[:,:nocc]
        orbo_j = kd.transform_single_mo_coeff(mo_coeff, kj_bz)[:,:nocc]

        for (ka_bz, kb_bz) in kab:
            kpts_a = kd.kpts[ka_bz]
            kpts_b = kd.kpts[kb_bz]
            ka = kd.bz2ibz[ka_bz]
            kb = kd.bz2ibz[kb_bz]
            orbv_a = kd.transform_single_mo_coeff(mo_coeff, ka_bz)[:,nocc:]
            orbv_b = kd.transform_single_mo_coeff(mo_coeff, kb_bz)[:,nocc:]
            oovv_ij[ka_bz] = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                    (kpts_i,kpts_a,kpts_j,kpts_b),
                                    compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nbzk
            nao2mo += 1

        for j in range(istart, iend):
            ka_bz = kijab[j][2]
            kb_bz = kijab[j][3]
            ka = kd.bz2ibz[ka_bz]
            kb = kd.bz2ibz[kb_bz]
            # Remove zero/padded elements from denominator
            eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

            eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
            t2_ijab = np.conj(oovv_ij[ka_bz]/eijab)
            if with_t2:
                t2[icount*2] = t2_ijab
                eib = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_ib = np.ix_(nonzero_opadding[ki], nonzero_vpadding[kb])
                eib[n0_ovp_ib] = (mo_e_o[ki][:,None] - mo_e_v[kb])[n0_ovp_ib]
                eja = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_ja = np.ix_(nonzero_opadding[kj], nonzero_vpadding[ka])
                eja[n0_ovp_ja] = (mo_e_o[kj][:,None] - mo_e_v[ka])[n0_ovp_ja]
                eijba = lib.direct_sum('ib,ja->ijba',eib,eja)
                t2_ijba = np.conj(oovv_ij[kb_bz]/eijba)
                t2[icount*2+1] = t2_ijba
            woovv = 2*oovv_ij[ka_bz] - oovv_ij[kb_bz].transpose(0,1,3,2)
            emp2 += np.einsum('ijab,ijab', t2_ijab, woovv).real * weight[icount] * nbzk**3
            icount += 1

    emp2 /= nbzk
    assert(icount == len(kijab))
    logger.debug(mp, "Number of ao2mo transformations performed in KMP2: %d", nao2mo)
    logger.timer(mp, 'KMP2', *t0)
    return emp2, t2

@lib.with_doc(kmp2.make_rdm1.__doc__)
def make_rdm1(mp, t2=None, kind="compact"):
    if kind not in ("compact", "padded"):
        raise ValueError("The 'kind' argument should be either 'compact' or 'padded'")
    d_imds = _gamma1_intermediates(mp, t2=t2)
    result = []
    padding_idxs = kmp2.padding_k_idx(mp, kind="joint")
    for (oo, vv), idxs in zip(zip(*d_imds), padding_idxs):
        oo += np.eye(*oo.shape)
        d = block_diag(oo, vv)
        d += d.conj().T
        if kind == "padded":
            result.append(d)
        else:
            result.append(d[np.ix_(idxs, idxs)])
    return result

def _gamma1_intermediates(mp, t2=None):
    if t2 is None:
        t2 = mp.t2
    if t2 is None:
        raise NotImplementedError("Run kmp2.kernel with `with_t2=True`")
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts_ibz = mp.nkpts
    dtype = t2.dtype

    dm1occ = np.zeros((nkpts_ibz, nocc, nocc), dtype=dtype)
    dm1vir = np.zeros((nkpts_ibz, nvir, nvir), dtype=dtype)

    kd = mp.kpts
    k_ibz = kd.ibz2bz
    kijab, weight, bz2ibz = kd.make_k4_ibz(sym='s2')
    nkpts = kd.nkpts
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]

                idx = ki*nkpts**2 + kj*nkpts + ka
                idx = bz2ibz[idx]

                if kb in k_ibz:
                    dm1vir[kd.bz2ibz[kb]] += einsum('ijax,ijay->yx', t2[idx*2].conj(), t2[idx*2]) * 2 -\
                                             einsum('ijax,ijya->yx', t2[idx*2].conj(), t2[idx*2+1])
                if kj in k_ibz:
                    dm1occ[kd.bz2ibz[kj]] += einsum('ixab,iyab->xy', t2[idx*2].conj(), t2[idx*2]) * 2 -\
                                             einsum('ixab,iyba->xy', t2[idx*2].conj(), t2[idx*2+1])
    return -dm1occ, dm1vir


class KsymAdaptedKMP2(kmp2.KMP2):
    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        if mo_energy is None: mo_energy = self.mo_energy
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            logger.warn('mo_coeff, mo_energy are not given.\n'
                        'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        # TODO: compute e_hf for non-canonical SCF
        self.e_hf = self._scf.e_tot

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose, with_t2=with_t2)
        logger.log(self, 'KMP2 energy = %.15g', self.e_corr)
        return self.e_corr, self.t2

    make_rdm1 = make_rdm1

KRMP2 = KMP2 = KsymAdaptedKMP2

from pyscf.pbc import scf
scf.khf_ksymm.KRHF.MP2 = lib.class_as_method(KRMP2)

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, mp

    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.build()

    kpts = cell.make_kpts([2,2,2], space_group_symmetry=True)
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mymp = mp.KMP2(kmf)
    emp2, t2 = mymp.kernel()
    print(emp2 - -0.13314158977189)
