import math
import numpy as np
import MDAnalysis as mda
import trimesh
import pyvista as pv


class ProteinSurface:
    def __init__(self, mda_atomgroup, solvent_rad):
        self.mda_atomgroup = mda_atomgroup
        self.solvent_rad = solvent_rad # FIXME private?
        
        # Generate surface
        dict_radius_to_voxel_sphere = {}
        grid_size = 0.7
        atom = self.mda_atomgroup[0]
        try:
            element = atom.element
        except mda.NoDataError:
            element = mda.topology.guessers.guess_atom_element(atom.type)
        vdw_rad = vdw_rads[element]
        sphere_rad = vdw_rad + solvent_rad
        next_sphere = trimesh.primitives.Sphere(center=atom.position, radius=sphere_rad, subdivisions=2)
        next_voxel = next_sphere.voxelized(grid_size) # FIXME corresponds to slow part
        voxel_points =list(next_voxel.points)
        min_x = next_voxel.origin[0]
        min_y = next_voxel.origin[1]
        min_z = next_voxel.origin[2]
        for atom in self.mda_atomgroup[1:]:
            try:
                element = atom.element
            except mda.NoDataError:
                element = mda.topology.guessers.guess_atom_element(atom.type)
            vdw_rad = vdw_rads[element]
            sphere_rad = vdw_rad + solvent_rad
            if sphere_rad not in dict_radius_to_voxel_sphere:
                next_sphere = trimesh.primitives.Sphere(center=[0,0,0], radius=sphere_rad, subdivisions=2)
                dict_radius_to_voxel_sphere[sphere_rad] = next_sphere.voxelized(grid_size)
            next_voxel = dict_radius_to_voxel_sphere[sphere_rad].copy()
            next_voxel.apply_translation(next_voxel.origin + atom.position)
            next_voxel = next_voxel.copy()
            voxel_points += list(next_voxel.points)
            min_x = min(min_x, next_voxel.origin[0])
            min_y = min(min_y, next_voxel.origin[1])
            min_z = min(min_z, next_voxel.origin[2])
        all_indices = trimesh.voxel.ops.points_to_indices(voxel_points, pitch=grid_size, origin=[min_x,min_y,min_z])
        self.surf = trimesh.voxel.VoxelGrid(trimesh.voxel.ops.sparse_to_matrix(all_indices))
        
        # The above code makes a surface with the correct shape.  But the surface is too large, and its origin
        # is [0,0,0].  The next few lines fix this.
        self.surf.apply_scale(grid_size)
        self.surf = self.surf.copy() # Necessary due to weird behavior (bug?) in trimesh library.
        # The round() code shifts the point to the nearest multiple of grid_size.  This is necessary
        # because otherwise the origin could be offset from the expected grid, breaking the boolean code.
        min_x = round(min_x / grid_size) * grid_size
        min_y = round(min_y / grid_size) * grid_size
        min_z = round(min_z / grid_size) * grid_size
        self.surf.apply_translation([min_x, min_y, min_z])
        self.surf = self.surf.copy()
        self.surf.fill()
        self.surf = self.surf.copy()

vdw_rads = {"C": 1.7, "H" : 1.2, "N" : 1.55, "O" : 1.52, "S" : 1.8}

def check_equal_pitches(voxel_grid_1, voxel_grid_2):
    """Verifies that voxel_grid_1 and voxel_grid_2 have
    pitches equal to each other and the same in each
    direction.  Raises an error if this is not true."""
    if len(np.unique(voxel_grid_1.pitch)) > 1:
        raise ValueError("voxel_grid_1.pitch values not uniform")
    if len(np.unique(voxel_grid_2.pitch)) > 1:
        raise ValueError("voxel_grid_2.pitch values not uniform")
    if not (voxel_grid_1.pitch == voxel_grid_2.pitch).all():
        raise ValueError("input VoxelGrid objects have different pitches")

def voxel_subtract(voxel_grid_1, voxel_grid_2):
    """voxel_grid_1 - voxel_grid_2. Returns a VoxelGrid
    object containing all points in voxel_grid_1 that are
    not in voxel_grid_2.  The returned VoxelGrid's origin is
    [0,0,0] and its pitch is the same as the input arguments'
    pitch.""" 
    
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
    pl.add_mesh(prot_pv, show_edges=True, opacity=0.5, color='grey')
    pl.add_mesh(pocket_pv, color="red")
    pl.subplot(0,1)
    pl.add_mesh(pocket_pv, color="red")
    pl.subplot(1,0)
    pl.add_mesh(prot_pv, show_edges=True, opacity=0.5, color='grey')
    pl.link_views()
    pl.show()
