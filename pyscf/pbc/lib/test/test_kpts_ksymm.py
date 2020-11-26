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

import unittest
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc.scf import khf, kuhf
from pyscf.pbc.lib import kpts as libkpts

class KnownValues(unittest.TestCase):
    def test_transform(self):
        cell = gto.Cell()
        cell.atom = """
            Si  0.0 0.0 0.0
            Si  1.3467560987 1.3467560987 1.3467560987
        """
        cell.a = [[0.0, 2.6935121974, 2.6935121974],
                  [2.6935121974, 0.0, 2.6935121974],
                  [2.6935121974, 2.6935121974, 0.0]]
        cell.verbose = 5
        cell.basis = 'gth-szv'
        cell.pseudo  = 'gth-pade'
        cell.build()
        kpts0 = cell.make_kpts([2,2,2])
        kmf = scf.KRKS(cell, kpts0)
        kmf.kernel()

        kpts = libkpts.make_kpts(cell, kpts0, space_group_symmetry=True, time_reversal_symmetry=True)
        dms_ibz = kmf.make_rdm1()[kpts.ibz2bz]
        dms_bz = kpts.transform_dm(dms_ibz)
        self.assertAlmostEqual(abs(dms_bz - kmf.make_rdm1()).max(), 0, 9)

        mo_coeff_ibz = np.asarray(kmf.mo_coeff)[kpts.ibz2bz]
        mo_coeff_bz = kpts.transform_mo_coeff(mo_coeff_ibz)
        dms_bz = khf.make_rdm1(mo_coeff_bz, kmf.mo_occ)
        self.assertAlmostEqual(abs(dms_bz - kmf.make_rdm1()).max(), 0, 9)

        mo_occ_ibz = np.asarray(kmf.mo_occ)[kpts.ibz2bz]
        mo_occ_bz = kpts.transform_mo_occ(mo_occ_ibz)
        self.assertAlmostEqual(abs(mo_occ_bz - np.asarray(kmf.mo_occ)).max(), 0, 9)

        mo_energy_ibz = np.asarray(kmf.mo_energy)[kpts.ibz2bz]
        mo_energy_bz = kpts.transform_mo_energy(mo_energy_ibz)
        self.assertAlmostEqual(abs(mo_energy_bz - np.asarray(kmf.mo_energy)).max(), 0 , 9)

        kmf = kmf.to_uhf()
        mo_coeff_ibz = []
        mo_coeff_ibz.append(np.asarray(kmf.mo_coeff[0])[kpts.ibz2bz])
        mo_coeff_ibz.append(np.asarray(kmf.mo_coeff[1])[kpts.ibz2bz])
        mo_coeff_bz = kpts.transform_mo_coeff(mo_coeff_ibz)
        dms_bz = kuhf.make_rdm1(mo_coeff_bz, kmf.mo_occ)
        self.assertAlmostEqual(abs(dms_bz - kmf.make_rdm1()).max(), 0, 9)

        mo_occ_ibz = []
        mo_occ_ibz.append(np.asarray(kmf.mo_occ[0])[kpts.ibz2bz])
        mo_occ_ibz.append(np.asarray(kmf.mo_occ[1])[kpts.ibz2bz])
        mo_occ_bz = kpts.transform_mo_occ(mo_occ_ibz)
        self.assertAlmostEqual(abs(mo_occ_bz - np.asarray(kmf.mo_occ)).max(), 0, 9)

        mo_energy_ibz = []
        mo_energy_ibz.append(np.asarray(kmf.mo_energy[0])[kpts.ibz2bz])
        mo_energy_ibz.append(np.asarray(kmf.mo_energy[1])[kpts.ibz2bz])
        mo_energy_bz = kpts.transform_mo_energy(mo_energy_ibz)
        self.assertAlmostEqual(abs(mo_energy_bz - np.asarray(kmf.mo_energy)).max(), 0 , 9)


if __name__ == "__main__":
    print("Tests for kpts")
    unittest.main()
