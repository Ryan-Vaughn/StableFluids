# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:05:00 2024

@author: vaugh
"""
import numpy as np
import cupy as cp
import pyvista as pv
import scipy.sparse as sps


class EulerianGrid3D:
    def __init__(self, shape, scale, center):
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
        self.scale = scale
        
        # Extracting 1d values from each 3d input.
        self.n_x, self.n_y, self.n_z = self.shape
        self.num_cells = self.n_x * self.n_y * self.n_z     
        self.del_x, self.del_y, self.del_z = self.scale
        
        # Primal Grid Variables
        self.p = np.zeros(shape) # Pressure
        self.d = np.zeros(shape) # Advected quantity (for visuals)
        
        self.div = np.zeros(shape) # divergence of velocity field

        # Dual Grid Variables
        self.u = np.zeros((self.n_x+1, self.n_y, self.n_z)) # x vel
        self.v = np.zeros((self.n_x, self.n_y+1, self.n_z)) # y vel
        self.w = np.zeros((self.n_x, self.n_y, self.n_z+1)) # z vel

        # Boundary masking variables for dual mesh.
        self.x_boundary = np.ones(self.u.shape).astype(bool)
        self.x_boundary[1:-1,:,:] = False

        self.y_boundary = np.ones(self.u.shape).astype(bool)
        self.y_boundary[:,1:-1,:] = False

        self.z_boundary = np.ones(self.u.shape).astype(bool)
        self.z_boundary[:,:,1:-1] = False
        
        # Length of each cell.
        self.l_x = self.del_x * self.n_x
        self.l_y = self.del_y * self.n_y
        self.l_z = self.del_z * self.n_z

        # Shortcut for indexing the grid.
        self.i, self.j, self.k = np.indices(self.shape)
        
        # Vars for interacting with the GUI.
        self.shape_bd = (self.n_x+1, self.n_y+1, self.n_z+1)   
        self.i_bd, self.j_bd, self.k_bd = np.indices(self.shape_bd)
        
        # self.laplace_mat = self.get_laplace_matrix()

    def partial_x(self, f):
        # The centered finite difference x partial derivative on the interior.
        f_x = (f[1:,:,:]- f[:-1,:,:]) / self.del_x 
        return f_x
    
    def partial_y(self, f):
        # The centered finite difference y partial derivative on interior.
        f_y = (f[:,1:,:]- f[:,:-1,:]) / self.del_y 
        return f_y
    
    def partial_z(self, f):
        # The centered finite difference z partial derivative on the interior.
        f_z = (f[:,:,1:]- f[:,:,:-1]) / self.del_z 
        return f_z
    
    def div(self, u, v, w):
        # The divergence operator that maps boundary to centers.
        u_x = self.partial_x(u)
        v_y = self.partial_y(v)
        w_z = self.partial_z(w)
        return u_x + v_y + w_z

    def get_laplace_matrix(self):
        # TODO: Create laplacian matrix.
        
        return self.laplace_mat

class StableFluidSolver:
    def __init__(self, grid, d_0, u_0, v_0, w_0, dt):
        self.grid = grid
        self.dt = dt
        
        self.d = d_0  # density
        
        self.u = u_0  # x - velocity
        self.v = v_0  # y - velocity
        self.w = w_0  # z - velocity
        
        self.div = np.zeros(grid.cent)
        self.p = np.zeros(grid.cent)
        
        # Masking variables for primal mesh.
        self.solid = None
        self.empty = None
        
        self.visc = 0
        self.diff_coeff = 0

        # self.diffusion = (sps.eye(self.grid.num_cells_bd)
        #                   - self.visc * self.dt * self.grid.laplace_mat)
        
        # self.laplacian = self.grid.laplace_mat
    
    def add_source(self, f, f_source):
        # method adds dt amount of the source grid to the grid.
        f = f + f_source * self.dt
        return f
    
    def diffuse(self, f):
        if len(f.shape) != 1:
            f = f.flatten()

        solution, _ = sps.linalg.bicg(self.diffusion, f)
        solution = solution.reshape(self.grid.shape_bd)
        return solution
    
    def advect(self, f_0, u, v, w, dt):
        f = np.zeros(self.grid.shape)
    
        # Backtrace by the vector field * dt
        x = self.grid.i_bd - dt * self.grid.n_x_bd / self.grid.l_x_bd * u
        y = self.grid.j_bd - dt * self.grid.n_y_bd / self.grid.l_y_bd * v
        z = self.grid.k_bd - dt * self.grid.n_z_bd / self.grid.l_z_bd * w
    
        # Linear interpolation setup
        x[x < 0.5] = 0.5
        y[y < 0.5] = 0.5
        z[z < 0.5] = 0.5
    
        x[x > self.grid.n_x + 0.5] = self.grid.n_x + 0.5
        y[y > self.grid.n_y + 0.5] = self.grid.n_y + 0.5
        z[z > self.grid.n_z + 0.5] = self.grid.n_z + 0.5
    
        i_0 = np.floor(x).astype(int)
        j_0 = np.floor(y).astype(int)
        k_0 = np.floor(z).astype(int)
        
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

    def compute_divergence(self, u, v, w):
        div = np.zeros(self.grid.shape_bd)
        div[self.grid.cent] = self.grid.div(u, v, w)
        return div
    
    def project(self, u, v, w):
        div = compute_divergence(u, v, w)

        # Solve the pressure Poisson equation
        q, _ = sps.linalg.bicg(self.laplacian, div)
        q = q.reshape(self.grid.shape_bd)

        # Take gradient of pressure (solution of PPE)
        q_x, q_y, q_z = self.grid.grad(q)

        # Correct velocity term using gradient of pressure. 
        u_proj = u - q_x
        v_proj = v - q_y
        w_proj = w - q_z
        
        return u_proj, v_proj, w_proj
    
    def set_dirichlet_boundary(self):

        self.u[0, :, :] = self.u[1, :, :]
        self.u[:, 0, :] = self.u[:, 1, :]
        self.u[:, :, 0] = self.u[:, :, 1]
        
        self.u[-1, :, :] = self.u[-2, :, :]
        self.u[:, -1, :] = self.u[:, -2, :]
        self.u[:, :, -1] = self.u[:, :, -2]
        
        self.v[0, :, :] = self.v[1, :, :]
        self.v[:, 0, :] = self.v[:, 1, :]
        self.v[:, :, 0] = self.v[:, :, 1]
        
        self.v[-1, :, :] = self.v[-2, :, :]
        self.v[:, -1, :] = self.v[:, -2, :]
        self.v[:, :, -1] = self.v[:, :, -2]
        
        self.w[0, :, :] = self.w[1, :, :]
        self.w[:, 0, :] = self.w[:, 1, :]
        self.w[:, :, 0] = self.w[:, :, 1]
        
        self.w[-1, :, :] = self.w[-2, :, :]
        self.w[:, -1, :] = self.w[:, -2, :]
        self.w[:, :, -1] = self.w[:, :, -2]

    def step(self):
        self.u = self.add_source(self.u, self.u_source)
        self.v = self.add_source(self.v, self.v_source)
        self.w = self.add_source(self.w, self.w_source)

        self.u = self.advect(self.u, self.u, self.v, self.w, self.dt)
        self.v = self.advect(self.v, self.u, self.v, self.w, self.dt)
        self.w = self.advect(self.w, self.u, self.v, self.w, self.dt)

        self.u = self.diffuse(self.u)
        self.v = self.diffuse(self.v)
        self.w = self.diffuse(self.w)
        
        self.u, self.v, self.w = self.project(self.u, self.v, self.w)

        self.d = self.add_source(self.d, self.d_source)
        self.d = self.advect(self.d, self.u, self.v, self.w, self.dt)
        self.d = self.diffuse(self.d)
        
        self.set_neumann_boundary()


if __name__ == "__main__":
    grid = EulerianGrid3D((3, 3, 3), (1/3, 1/3, 1/3), (1,1,1))

    d_0 = 1 * np.ones(grid.shape)

    u_0 = np.zeros(grid.u.shape)
    v_0 = np.zeros(grid.v.shape)
    w_0 = np.zeros(grid.w.shape)
    
    u_0[1,1,1] = 1
    v_0[1,1,1] = 1
    w_0[1,1,1] = 1
    dt = .001
    
    plot_test = np.zeros((3,3,3)) 
    plot_test[1,1,1] = 10

    solver = StableFluidSolver(grid, d_0, u_0, v_0, w_0, dt)