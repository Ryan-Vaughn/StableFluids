# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:05:00 2024

@author: vaugh
"""
import numpy as np
import cupy as cp
import scipy.sparse as sps


class FDGrid3D:
    def __init__(self, shape, dimensions, center, dim=3):
        """
        Class for a simple cubical grid for implementing the finite difference
        method.

        Parameters
        ----------
        shape : tuple[int]
            The number of interior nodes in each dimension of the mesh.
        dimensions : tuple[float]
            The length in Euclidean distance that the interior of the mesh
            corresponds to.
        center : tuple[int]
            The indices of the grid point that corresponds to the Euclidean
            origin.

        Returns
        -------
        None.

        """
        self.shape = shape
        self.dimensions = dimensions
        
        # Extracting 1d values from each 3d input.
        self.n_x, self.n_y, self.n_z = self.shape
        self.num_cells = self.n_x * self.n_y * self.n_z

        self.l_x, self.l_y, self.l_z = self.dimensions
        
        # Length of each cell.
        self.del_x = self.l_x / self.n_x
        self.del_y = self.l_y / self.n_y
        self.del_z = self.l_z / self.n_z
        
        # A boundary cell is placed on each end to account for boundary conds.
        # These parameters keep track of cell data including boundary.
        self.n_x_bd = self.n_x + 2
        self.n_y_bd = self.n_y + 2
        self.n_z_bd = self.n_z + 2

        self.num_cells_bd = self.n_x_bd * self.n_y_bd * self.n_z_bd

        self.l_x_bd = self.l_x + 2 * self.del_x
        self.l_y_bd = self.l_y + 2 * self.del_y
        self.l_z_bd = self.l_z + 2 * self.del_z

        # Shortcut for indexing the grid.
        self.i, self.j, self.k = np.indices(self.shape)
        self.i_bd, self.j_bd, self.k_bd = np.indices(self.shape_bd)

        # Shortcut for indexing forward and backward shifts.
        self.cent = np.s_[1:self.n_x_bd-1, 1:self.n_y_bd-1, 1:self.n_z_bd-1]        

        self.x_b = np.s_[0:self.n_x, 1:self.n_y_bd-1, 1:self.n_z_bd-1]
        self.x_f = np.s_[2:self.n_x_bd, 1:self.n_y_bd-1, 1:self.n_z_bd-1]
        self.y_b = np.s_[1:self.n_x_bd-1, 0:self.n_y, 1:self.n_z_bd-1]
        self.y_f = np.s_[1:self.n_x_bd-1, 2:self.n_y_bd, 1:self.n_z_bd-1]
        self.z_b = np.s_[1:self.n_x_bd-1, 1:self.n_y_bd-1, 0:self.n_z]
        self.z_f = np.s_[1:self.n_x_bd-1, 1:self.n_y_bd-1, 2:self.n_z_bd]
        
    def thicken(self, f):
        # If f is in flat form, reshape to the grid shape.

        if f.shape == self.num_cells_bd:
            f = f.reshape(self.shape_bd)  # Should almost always be this size.
        if f.shape == self.num_cells:
            f = f.reshape(self.shape)
        else:
            print("Unexpected length of function.")


    def partial_x(self, f):
        # The centered finite difference x partial derivative on the interior.
        f_x = ((f[self.x_f] + f[self.x_b] - 2 * f[self.cent])
               / self.del_x ** 2)
        return f_x
    
    def partial_y(self, f):
        # The centered finite difference y partial derivative on interior.
        f_y = ((f[self.y_f] + f[self.y_b] - 2 * f[self.cent])
               / self.del_y ** 2)
        return f_y
    
    def partial_z(self, f):
        # The centered finite difference z partial derivative on the interior.
        f_z = ((f[self.z_f] + f[self.z_b] - 2 * f[self.cent])
               / self.del_z ** 2)
        return f_z
    
    def grad(self, f):
        # The gradient on the interior
        f_x = self.partial_x(f)
        f_y = self.partial_y(f)
        f_z = self.partial_z(f)
       
        return f_x, f_y, f_z
    
    def div(self, u, v, w):
        # The divergence on the interior.
        u_x = self.partial_x(u)
        v_y = self.partial_y(v)
        w_z = self.partial_z(w)
        return u_x + v_y + w_z

    def get_diffusion_matrix(self):

        no_shift = sps.eye(self.num_cells_bd)

        shift_x_up = sps.eye(self.num_cells_bd, k=self.n_y_bd*self.n_z_bd)
        shift_x_down = sps.eye(self.num_cells_bd, k=-self.n_y_bd*self.n_z_bd)

        y_diag_tile = ([1] * (self.n_y_bd - 1) * self.n_z_bd + 
                       [0] * self.n_z_bd)
        shift_y_diag = np.tile(y_diag_tile, self.n_x_bd)
        shift_y_diag = shift_y_diag[:-self.n_z_bd]
        shift_y_up = sps.diags(shift_y_diag, offsets=self.n_z_bd)
        shift_y_down = sps.diags(shift_y_diag, offsets=-self.n_z_bd)

        z_diag_tile = [1] * (self.n_z_bd - 1) + [0]
        shift_z_diag = np.tile(z_diag_tile, self.n_y_bd*self.n_z_bd)
        shift_z_diag = shift_z_diag[:-1]
        shift_z_up = sps.diags(shift_z_diag, offsets=1)
        shift_z_down = sps.diags(shift_z_diag, offsets=-1)
        
        self.diff_mat = (-6 * no_shift 
                         + shift_x_up 
                         + shift_x_down 
                         + shift_y_up 
                         + shift_y_down 
                         + shift_z_down 
                         + shift_z_up)
        return self.diff_mat

    def add_source(source, dt):
        # method adds dt amount of the source grid to the grid.
        pass

    def diffuse(dens, dens_prev, dt, diff):
        pass

    def advect(self, f_0, u, v, w, dt):
        f = np.zeros(self.shape)

        # Backtrace by the vector field * dt
        x = self.i_bd - dt * self.n_x_bd / self.l_x_bd * u
        y = self.j_bd - dt * self.n_y_bd / self.l_y_bd * v
        z = self.k_bd - dt * self.n_z_bd / self.l_z_bd * w

        # Linear interpolation setup
        x[x < 0.5] = 0.5
        y[y < 0.5] = 0.5
        z[z < 0.5] = 0.5

        x[x > self.n_x + 0.5] = self.n_x + 0.5
        y[y > self.n_y + 0.5] = self.n_y + 0.5
        z[z > self.n_z + 0.5] = self.n_z + 0.5

        i_0 = np.floor(x)
        j_0 = np.floor(y)
        k_0 = np.floor(z)
        
        i_1 = i_0 + 1
        j_1 = j_0 + 1
        k_1 = k_0 + 1
        
        r_1 = x - i_0
        r_0 = 1 - r_1
        s_1 = y - j_0
        s_0 = 1 - s_1
        t_1 = z - k_0
        t_0 = 1 - t_1
        
        # Linearly interpolate the value of f at each point from the backtrace.
        f = (r_0 * (s_0 * (t_0 * f_0[i_0, j_0, k_0] 
                           + t_1 * f_0[i_0, j_0, k_1])
                    + s_1 * (t_0 * f_0[i_0, j_1, k_0] 
                             + t_1 * f_0[i_0, i_1, j_1]))
             + r_1 * (s_0 * (t_0 * f_0[i_1, j_0, k_0]
                             + t_1 * f_0[i_1, i_0, k_1]) 
                      + s_1 * (t_0 * f_0[i_1, j_1, k_0]
                               + t_1 * f_0[i_1, i_1, j_1])))
        return f
        
    def project(grid):
        pass
    
    def set_boundary()
        pass
def density_iteration(source, dens, dens_prev, dt, diff):
    pass

if __name__ == "__main__":
    grid = Grid((3, 3, 3), (3, 3, 3))