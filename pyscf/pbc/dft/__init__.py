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

from pyscf.pbc.dft.gen_grid import UniformGrids, BeckeGrids
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import uks
from pyscf.pbc.dft import roks
from pyscf.pbc.dft import krks
from pyscf.pbc.dft import kuks
from pyscf.pbc.dft import kroks
from pyscf.pbc.dft import krks_ksymm
from pyscf.pbc.dft import kuks_ksymm
from pyscf.pbc.dft import krkspu
from pyscf.pbc.dft import kukspu

UKS = uks.UKS
ROKS = roks.ROKS

#KRKS = krks.KRKS
def KRKS(cell, *args, **kwargs):
    for arg in args:
        if hasattr(arg, "kpts_ibz"):
            return krks_ksymm.KRKS(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if hasattr(kwargs['kpts'], "kpts_ibz"):
            return krks_ksymm.KRKS(cell, *args, **kwargs)
    return krks.KRKS(cell, *args, **kwargs)

#KUKS = kuks.KUKS
def KUKS(cell, *args, **kwargs):
    for arg in args:
        if hasattr(arg, "kpts_ibz"):
            return kuks_ksymm.KUKS(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if hasattr(kwargs['kpts'], "kpts_ibz"):
            return kuks_ksymm.KUKS(cell, *args, **kwargs)
    return kuks.KUKS(cell, *args, **kwargs)

KROKS = kroks.KROKS

KRKSpU = krkspu.KRKSpU
KUKSpU = kukspu.KUKSpU

def RKS(cell, *args, **kwargs):
    if cell.spin == 0:
        return rks.RKS(cell, *args, **kwargs)
    else:
        return roks.ROKS(cell, *args, **kwargs)
RKS.__doc__ = rks.RKS.__doc__

def KS(cell, *args, **kwargs):
    if cell.spin == 0:
        return rks.RKS(cell, *args, **kwargs)
    else:
        return uks.UKS(cell, *args, **kwargs)
KS.__doc__ = '''
A wrap function to create DFT object (RKS or UKS) for PBC systems.\n
''' + rks.RKS.__doc__

def KKS(cell, *args, **kwargs):
    if cell.spin == 0:
        return krks.KRKS(cell, *args, **kwargs)
    else:
        return kuks.KUKS(cell, *args, **kwargs)
KKS.__doc__ = '''
A wrap function to create DFT object with k-point sampling (KRKS or KUKS).\n
''' + krks.KRKS.__doc__
