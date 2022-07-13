import math
import numpy as np
import MDAnalysis as mda
import trimesh
import pyvista as pv

class AtomGeo:
    def __init__(self, mda_atom, voxel_sphere):
        self.mda_atom = mda_atom
        self.voxel_sphere = voxel_sphere


class ProteinSurface:
    def __init__(self, mda_atomgroup, solvent_rad=1.4, grid_size=0.7):
        self.mda_atomgroup = mda_atomgroup
        self.solvent_rad = solvent_rad # FIXME private?
        
        # Generate surface
        dict_radius_to_voxel_sphere = {}
        atom = self.mda_atomgroup[0]
        try:
            element = atom.element
        except mda.NoDataError:
            element = mda.topology.guessers.guess_atom_element(atom.type)
        vdw_rad = vdw_rads[element]
        sphere_rad = vdw_rad + solvent_rad
        next_sphere = trimesh.primitives.Sphere(center=atom.position, radius=sphere_rad, subdivisions=2) # FIXME wrong center?
        next_voxel = next_sphere.voxelized(grid_size)
        voxel_points =list(next_voxel.points)
        min_x = next_voxel.origin[0]
        min_y = next_voxel.origin[1]
        min_z = next_voxel.origin[2]
        self.atom_geo_list = []
        for atom in self.mda_atomgroup[1:]:
            try:
                element = atom.element
            except mda.NoDataError:
                element = mda.topology.guessers.guess_atom_element(atom.type)
            vdw_rad = vdw_rads[element]
            sphere_rad = vdw_rad + solvent_rad
            if sphere_rad not in dict_radius_to_voxel_sphere:
                next_sphere = trimesh.primitives.Sphere(center=[0,0,0], radius=sphere_rad-3/5*grid_size, subdivisions=2)
                dict_radius_to_voxel_sphere[sphere_rad] = next_sphere.voxelized(grid_size)
            next_voxel = dict_radius_to_voxel_sphere[sphere_rad].copy()
            # The round() code shifts the point to the nearest multiple of grid_size.  This is necessary
            # because otherwise the origin could be offset from the expected grid, breaking the boolean code.
            exact_trans = atom.position
            approx_trans_x = round(exact_trans[0] / grid_size) * grid_size
            approx_trans_y = round(exact_trans[1] / grid_size) * grid_size
            approx_trans_z = round(exact_trans[2] / grid_size) * grid_size

            next_voxel.apply_translation(np.array([approx_trans_x, approx_trans_y, approx_trans_z]))
            next_voxel = next_voxel.copy()
            voxel_points += list(next_voxel.points)
            min_x = min(min_x, next_voxel.origin[0])
            min_y = min(min_y, next_voxel.origin[1])
            min_z = min(min_z, next_voxel.origin[2])
            self.atom_geo_list.append(AtomGeo(atom, next_voxel))
        all_indices = trimesh.voxel.ops.points_to_indices(voxel_points, pitch=grid_size, origin=[min_x,min_y,min_z])
        self.surf = trimesh.voxel.VoxelGrid(trimesh.voxel.ops.sparse_to_matrix(all_indices))
        
        # The above code makes a surface with the correct shape.  But the surface is too large, and its origin
        # is [0,0,0].  The next few lines fix this.
        self.surf.apply_scale(grid_size)
        self.surf = self.surf.copy() # Necessary due to weird behavior (bug?) in trimesh library.
        self.surf.apply_translation([min(np.array(voxel_points)[:,0]), min(np.array(voxel_points)[:,1]), min(np.array(voxel_points)[:,2])])
        self.surf = self.surf.copy()
        self.surf.fill()
        self.surf = self.surf.copy()

vdw_rads = {"C": 1.7, "H" : 1.2, "N" : 1.55, "O" : 1.52, "S" : 1.8}

def check_equal_pitches(voxel_grid_1, voxel_grid_2):
    """
    Verify that two voxel grids have the same pitch as each other.
    
    Verifies that voxel_grid_1 and voxel_grid_2 have
    pitches equal to each other and the same in each
    direction.  Raises an error if this is not true.
    
    Parameters
    ----------
    voxel_grid_1 : trimesh VoxelGrid object
        First shape to compare
    voxel_grid_2 : trimesh VoxelGrid object
        Second shape to compare

    Returns
    -------
    bool
        Whether the two shapes have the same pitch as each other and in all directions
    """

    if len(np.unique(voxel_grid_1.pitch)) > 1:
        raise ValueError("voxel_grid_1.pitch values not uniform")
    if len(np.unique(voxel_grid_2.pitch)) > 1:
        raise ValueError("voxel_grid_2.pitch values not uniform")
    if not (voxel_grid_1.pitch == voxel_grid_2.pitch).all():
        raise ValueError("input VoxelGrid objects have different pitches")

def voxel_subtract(voxel_grid_1, voxel_grid_2):
    """voxel_grid_1 - voxel_grid_2. Returns a VoxelGrid
    object containing all points in voxel_grid_1 that are
    not in voxel_grid_2.  The returned VoxelGrid's pitch is
    the same as the input arguments' pitch.""" 
    
    check_equal_pitches(voxel_grid_1, voxel_grid_2)
    
    # Check which points from voxel_grid_1 are in voxel_grid_2.
    vox_1_without_2_points = [] # initialization
    # FIXME decimal place is magic number
    vox_2_points = set(tuple(point) for point in voxel_grid_2.points.round(decimals=5).tolist())
    for point in voxel_grid_1.points.round(decimals=5).tolist():
        if tuple(point) not in vox_2_points:
            vox_1_without_2_points.append(point)

    vox_1_without_2_points = np.array(vox_1_without_2_points)
    min_x = min(vox_1_without_2_points[:,0])
    min_y = min(vox_1_without_2_points[:,1])
    min_z = min(vox_1_without_2_points[:,2])
    vox_1_without_2_indices = trimesh.voxel.ops.points_to_indices(vox_1_without_2_points,
                                                                  pitch=voxel_grid_1.pitch[0],
                                                                  origin=[min_x,min_y,min_z])
    vox_1_without_2_matrix = trimesh.voxel.ops.sparse_to_matrix(vox_1_without_2_indices)
    vox_1_without_2_voxel_grid = trimesh.voxel.VoxelGrid(vox_1_without_2_matrix)
    vox_1_without_2_voxel_grid.apply_scale(voxel_grid_1.pitch[0])
    vox_1_without_2_voxel_grid = vox_1_without_2_voxel_grid.copy()
    vox_1_without_2_voxel_grid.apply_translation([min_x, min_y, min_z])
    vox_1_without_2_voxel_grid = vox_1_without_2_voxel_grid.copy()
    return vox_1_without_2_voxel_grid

def voxel_or(voxel_grid_1, voxel_grid_2):
    """Returns a VoxelGrid object containing all points in
    voxel_grid_1 and/or voxel_grid_2.  Unlike the trimesh library's
    boolean_sparse function, this function does not require that the
    two input grids have the same shape."""
    
    check_equal_pitches(voxel_grid_1, voxel_grid_2)
    vox_1_or_2_points = np.append(voxel_grid_1.points, voxel_grid_2.points, axis=0)
    min_x = min(vox_1_or_2_points[:,0])
    min_y = min(vox_1_or_2_points[:,1])
    min_z = min(vox_1_or_2_points[:,2])
    vox_1_or_2_indices = trimesh.voxel.ops.points_to_indices(vox_1_or_2_points,
                                                                  pitch=voxel_grid_1.pitch[0],
                                                                  origin=[min_x, min_y, min_z])
    vox_1_or_2_matrix = trimesh.voxel.ops.sparse_to_matrix(vox_1_or_2_indices)
    vox_1_or_2_voxel_grid = trimesh.voxel.VoxelGrid(vox_1_or_2_matrix)
    vox_1_or_2_voxel_grid.apply_scale(voxel_grid_1.pitch[0])
    vox_1_or_2_voxel_grid = vox_1_or_2_voxel_grid.copy()
    vox_1_or_2_voxel_grid.apply_translation([min_x, min_y, min_z])
    vox_1_or_2_voxel_grid = vox_1_or_2_voxel_grid.copy()
    return vox_1_or_2_voxel_grid

def voxel_and(voxel_grid_1, voxel_grid_2):
    """Returns a VoxelGrid
    object containing all points in both voxel_grid_1 and
    voxel_grid_2.  Returns None if the two objects have no
    points in common.""" 
    
    check_equal_pitches(voxel_grid_1, voxel_grid_2)
    
    # Check which points from voxel_grid_1 are in voxel_grid_2.
    vox_1_and_2_points = [] # initialization
    # FIXME decimal place is magic number
    vox_2_points = set(tuple(point) for point in voxel_grid_2.points.round(decimals=5).tolist())
    for point in voxel_grid_1.points.round(decimals=5).tolist():
        if tuple(point) in vox_2_points:
            vox_1_and_2_points.append(point)

    if len(vox_1_and_2_points) == 0:
        return None
    vox_1_and_2_points = np.array(vox_1_and_2_points)
    min_x = min(vox_1_and_2_points[:,0])
    min_y = min(vox_1_and_2_points[:,1])
    min_z = min(vox_1_and_2_points[:,2])
    vox_1_and_2_indices = trimesh.voxel.ops.points_to_indices(vox_1_and_2_points,
                                                              pitch=voxel_grid_1.pitch[0],
                                                              origin=[min_x,min_y,min_z])
    vox_1_and_2_matrix = trimesh.voxel.ops.sparse_to_matrix(vox_1_and_2_indices)
    vox_1_and_2_voxel_grid = trimesh.voxel.VoxelGrid(vox_1_and_2_matrix)
    vox_1_and_2_voxel_grid.apply_scale(voxel_grid_1.pitch[0])
    vox_1_and_2_voxel_grid = vox_1_and_2_voxel_grid.copy()
    vox_1_and_2_voxel_grid.apply_translation([min_x, min_y, min_z])
    vox_1_and_2_voxel_grid = vox_1_and_2_voxel_grid.copy()
    return vox_1_and_2_voxel_grid

def show_pocket(prot_vox, pocket_vox):

    prot_vox = prot_vox.copy()
    prot_vox.hollow()
    prot_trimesh = prot_vox.as_boxes()
    prot_pv = pv.wrap(prot_trimesh)

    pocket_vox = pocket_vox.copy()
    pocket_vox.hollow()
    pocket_trimesh = pocket_vox.as_boxes()
    pocket_pv = pv.wrap(pocket_trimesh)
    
    pl = pv.Plotter(shape=(2,2))
    pl.add_mesh(prot_pv, color='red')
    pl.add_mesh(pocket_pv, color="blue")
    pl.subplot(0,1)
    pl.add_mesh(pocket_pv, color="blue")
    pl.subplot(1,0)
    pl.add_mesh(prot_pv, color='red')
    pl.link_views()
    pl.show()