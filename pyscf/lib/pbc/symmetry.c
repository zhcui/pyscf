#include "stdio.h"

void symmetrize(double* rhoR_b, double* rhoR_a, int* op, int* mesh)
{
  int nx = mesh[0];
  int ny = mesh[1];
  int nz = mesh[2];

  #pragma omp for schedule(static)
    for (int x = 0; x < nx; x++)
      for (int y = 0; y < ny; y++)
        for (int z = 0; z < nz; z++) {
            int x_p = ((op[0] * x + op[3] * y + op[6] * z) % nx + nx) % nx;
            int y_p = ((op[1] * x + op[4] * y + op[7] * z) % ny + ny) % ny;
            int z_p = ((op[2] * x + op[5] * y + op[8] * z) % nz + nz) % nz;
            rhoR_b[x_p*ny*nz + y_p*nz + z_p] += rhoR_a[x*ny*nz + y*nz + z];
        }
}


