import math
import numpy as np
import MDAnalysis as mda
import trimesh
import pyvista as pv
import scipy

class AtomGeo:
    """A surface representation of an atom.

    Parameters
    ----------
    mda_atom : MDAnalysis Atom
        The atom of interest.
    voxel_sphere : trimesh VoxelGrid
        The sphere of volume occupied by the atom.  When constructed as
        part of a ProteinSurface, this will have the radius of a solvent
        molecule added to it.

    Attributes
    ----------
    mda_atom : MDAnalysis Atom
        The atom that was passed during object construction.
    voxel_sphere : trimesh VoxelGrid
        The sphere that was passed during object construction.
    """
    def __init__(self, mda_atom, voxel_sphere):
        self.mda_atom = mda_atom
        self.voxel_sphere = voxel_sphere


class ProteinSurface:
    """A surface representation of a protein.

    Parameters
    ----------
    mda_atomgroup : MDAnalysis Atomgroup
        All the atoms to be included in the surface.
    solvent_rad : float, optional
        The protein surface is constructed from the protein atoms'
        van der Waals radii plus the radius of a hypothetical solvent molecule.
        The default value is 1.09 (the van der Waals radius of hydrogen); this was
        chosen because POVME uses this value.
    grid_size : float, optional
        The length (in Angstroms) of each side of a voxel.  The default
        value is 0.7.

    Attributes
    ----------
    surf : trimesh VoxelGrid
        The surface of the protein.  The interior is filled.
    mda_atomgroup : MDAnalysis Atomgroup
        The AtomGroup that was provided during ProteinSurface construction.
    atom_geo_list : list of AtomGeo objects
        Contains an AtomGeo for each atom in mda_atomgroup.
    """
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
        next_voxel = generate_voxelized_sphere(sphere_rad, atom.position, grid_size)
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
                next_sphere = generate_voxelized_sphere(sphere_rad, [0,0,0], grid_size)
                dict_radius_to_voxel_sphere[sphere_rad] = next_sphere
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


def get_pocket_atoms(protein_surface_obj, pocket_surf):
    """
    Get all protein atoms that border the pocket.
    
    Parameters
    ----------
    protein_surface_obj : ProteinSurface
        The protein molecule.  The object must be
        the ProteinSurface type defined in this software.
    pocket_surf : trimesh VoxelGrod
        The pocket.

    Returns
    -------
    list of AtomGeo objects
        An AtomGeo for each atom bordering the pocket.
    trimesh VoxelGrid object
        A surface containing all atoms bordering the pocket.
    """
    pocket_atoms = []
    for atom_geo in protein_surface_obj.atom_geo_list:
        atom_dilated = dilate_voxel_grid(atom_geo.voxel_sphere)
        atom_pocket_overlap = voxel_and(atom_dilated, pocket_surf)
        if atom_pocket_overlap:
            pocket_atoms.append(atom_geo)
    pocket = pocket_atoms[0].voxel_sphere
    for atom_geo in pocket_atoms[1:]:
        pocket = voxel_or(pocket, atom_geo.voxel_sphere)
    return pocket_atoms, pocket


def write_voxels_to_pdb(voxel_grid, pdb_filename):
    """
    Write a VoxelGrid object to a PDB file.

    Each voxel is written as an atom of name "X".  The result can
    be viewed in PyMOL; users may want to use `set sphere_scale [radius]`
    where `radius` is 1/2 the voxel grid size.
    
    Parameters
    ----------
    voxel_grid : trimesh VoxelGrid object
        The object to be written.
    pdb_filename : string ending in ".pdb"
        The file to write.
    """
    universe_voxel = mda.Universe.empty(len(voxel_grid.points), trajectory=True)
    universe_voxel.atoms.positions = voxel_grid.points
    universe_voxel.atoms.write(pdb_filename)


def get_prot_pocket(protein_surf, pocket_surf):
    """
    Gets all voxels of the protein that border the pocket.
    
    Parameters
    ----------
    protein_surf : trimesh VoxelGrid object
        The protein
    pocket_surf : trimesh VoxelGrid object
        The pocket

    Returns
    -------
    trimesh VoxelGrid object
        A VoxelGrid containing voxels in protein_surf that border protein_surf
    """
    dilated_pocket = dilate_voxel_grid(pocket_surf)
    return voxel_and(protein_surf, dilated_pocket)


def dilate_voxel_grid(voxel_grid):
    """
    Creates a dilated version of the input shape.
    
    The function creates a version of the input shape that has
    been expanded by 1 grid position in all dimensions.
    
    Parameters
    ----------
    voxel_grid : trimesh VoxelGrid object
        The input shape

    Returns
    -------
    trimesh VoxelGrid object
        The dilated shape
    """
    # The function doesn't know whether the outer edges of
    # voxel_grid contain any filled points.  If they do,
    # scipy's binary_dilation won't expand the input matrix's
    # size to fit the enlarged shape.  This is fixed by adding
    # an extra row and column to the matrix before dilating.
    big_array = np.pad(voxel_grid.matrix, pad_width=((1,0), (1,0), (1,0)), constant_values=False)
    dilated_x_orig = voxel_grid.origin[0] - voxel_grid.scale[0]
    dilated_y_orig = voxel_grid.origin[1] - voxel_grid.scale[1]
    dilated_z_orig = voxel_grid.origin[2] - voxel_grid.scale[2]
    dilated_matrix = scipy.ndimage.binary_dilation(big_array)

    dilated_vg = trimesh.voxel.VoxelGrid(dilated_matrix)
    dilated_vg.apply_scale(0.5)
    dilated_vg.apply_translation([dilated_x_orig, dilated_y_orig, dilated_z_orig])
    dilated_vg = dilated_vg.copy()
    return dilated_vg


def generate_voxelized_sphere(radius, center, grid_size):
    """
    Creates a voxelized sphere.
    
    The function creates a trimesh VoxelGrid sphere
    based on the input parameters.
    
    Parameters
    ----------
    radius : float
        Radius of the sphere
    center : list-like object containing three floats
        x, y, z coordinates of the sphere's center.
    grid_size : float
        Length of a side of the voxel grid.  0.5 is a reasonable
        choice.

    Returns
    -------
    trimesh VoxelGrid object
        A VoxelGrid containing the sphere.
    """
    x_min = round((center[0]-radius) / grid_size) * grid_size
    x_max = round((center[0]+radius) / grid_size) * grid_size
    y_min = round((center[1]-radius) / grid_size) * grid_size
    y_max = round((center[1]+radius) / grid_size) * grid_size
    z_min = round((center[2]-radius) / grid_size) * grid_size
    z_max = round((center[2]+radius) / grid_size) * grid_size
    
    # Create a list of all points in the grid.
    # See https://stackoverflow.com/a/12891609
    # The complex numbers cause numpy to go from the min to max values
    # (inclusive of both) with the number of points equal to the complex
    # number.
    X, Y, Z = np.mgrid[x_min:x_max:complex(0, (x_max-x_min)/grid_size+1),
                       y_min:y_max:complex(0, (y_max-y_min)/grid_size+1),
                       z_min:z_max:complex(0, (z_max-z_min)/grid_size+1)]
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    all_grid_points = positions.T

    # Get all points within sphere.
    # First construct a list of booleans for whether each point is in sphere.
    # The ending [:,0] is because numpy gives a list-of-lists where each inner list
    # has length 1.
    is_in_sphere = (scipy.spatial.distance.cdist(all_grid_points, [center]) < radius)[:,0]
    points_in_sphere = all_grid_points[is_in_sphere]

    # Convert list of points to trimesh VoxelGrid object.
    all_indices = trimesh.voxel.ops.points_to_indices(points_in_sphere, pitch=grid_size,
                                                      origin=[min(points_in_sphere[:,0]), min(points_in_sphere[:,1]),
                                                              min(points_in_sphere[:,2])])
    sphere_voxel = trimesh.voxel.VoxelGrid(trimesh.voxel.ops.sparse_to_matrix(all_indices))
    sphere_voxel.apply_scale(grid_size)
    sphere_voxel = sphere_voxel.copy()
    sphere_voxel.apply_translation([min(points_in_sphere[:,0]), min(points_in_sphere[:,1]), min(points_in_sphere[:,2])])
    sphere_voxel = sphere_voxel.copy()
    return sphere_voxel


def generate_voxelized_box(lengths, center, grid_size):
    """
    Creates a voxelized box.
    
    The function creates a trimesh VoxelGrid box
    based on the input parameters.
    
    Parameters
    ----------
    lengths : list-like object containing three floats
        The box's side lengths in x, y, z directions 
    center : list-like object containing three floats
        x, y, z coordinates of the sphere's center.
    grid_size : float
        Length of a side of the voxel grid.  0.5 is a reasonable
        choice.

    Returns
    -------
    trimesh VoxelGrid object
        A VoxelGrid containing the box.
    """

    # The code works with the center points of each voxel, but the function's output box
    # extends to the voxel's edges.  The 0.5*grid_size accounts for this; the goal is to find
    # the voxel whose edge is closest to that specified by the input parameters.
    x_min = round((center[0]-0.5*lengths[0]+0.5*grid_size) / grid_size) * grid_size
    x_max = round((center[0]+0.5*lengths[0]-0.5*grid_size) / grid_size) * grid_size
    y_min = round((center[1]-0.5*lengths[1]+0.5*grid_size) / grid_size) * grid_size
    y_max = round((center[1]+0.5*lengths[1]-0.5*grid_size) / grid_size) * grid_size
    z_min = round((center[2]-0.5*lengths[2]+0.5*grid_size) / grid_size) * grid_size
    z_max = round((center[2]+0.5*lengths[2]-0.5*grid_size) / grid_size) * grid_size
    
    # Create a list of all points in the grid.
    # See https://stackoverflow.com/a/12891609
    # The complex numbers cause numpy to go from the min to max values
    # (inclusive of both) with the number of points equal to the complex
    # number.
    X, Y, Z = np.mgrid[x_min:x_max:complex(0, (x_max-x_min)/grid_size+1),
                       y_min:y_max:complex(0, (y_max-y_min)/grid_size+1),
                       z_min:z_max:complex(0, (z_max-z_min)/grid_size+1)]
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    points_in_box = positions.T
    
    all_indices = trimesh.voxel.ops.points_to_indices(points_in_box, pitch=grid_size,
                                                      origin=[min(points_in_box[:,0]),
                                                              min(points_in_box[:,1]),
                                                              min(points_in_box[:,2])])
    box_voxel = trimesh.voxel.VoxelGrid(trimesh.voxel.ops.sparse_to_matrix(all_indices))
    box_voxel.apply_scale(grid_size)
    box_voxel = box_voxel.copy()
    box_voxel.apply_translation([min(points_in_box[:,0]),
                                 min(points_in_box[:,1]),
                                 min(points_in_box[:,2])])
    box_voxel = box_voxel.copy()
    return box_voxel


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
    """
    voxel_grid_1 - voxel_grid_2.
    
    Returns a VoxelGrid object containing all points in
    voxel_grid_1 that are not in voxel_grid_2.  The
    returned VoxelGrid's pitch is
    the same as the input arguments' pitch.
    
    Parameters
    ----------
    voxel_grid_1 : trimesh VoxelGrid object
        Shape to be subtracted from.  Must have same pitch in
        each dimension.
    voxel_grid_2 : trimesh VoxelGrid object
        Shape that is subtracted away.  Must have same pitch
        in each dimension (and same pitch as voxel_grid_1).

    Returns
    -------
    trimesh VoxelGrid object
        A VoxelGrid with the remaining points from
        voxel_grid_1
    """ 
    
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
    """Finds all points in voxel_grid_1 and/or voxel_grid_2.
    
    Returns a VoxelGrid with all points in at least one of voxel_grid_1 and
    voxel_grid_2.  Unlike the trimesh library's
    boolean_sparse function, this function does not require that the
    two input grids have the same shape.
    
    Parameters
    ----------
    voxel_grid_1 : trimesh VoxelGrid object
        First shape to be combined.  Must have same pitch in
        each dimension.
    voxel_grid_2 : trimesh VoxelGrid object
        Second shape to be combined.  Must have same pitch
        in each dimension (and same pitch as voxel_grid_1).

    Returns
    -------
    trimesh VoxelGrid object
        A VoxelGrid with all points from voxel_grid_1 and/or voxel_grid_2
    """
    
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
    """Finds all points in both voxel_grid_1 and voxel_grid_2.
    
    Returns a VoxelGrid with all points in both voxel_grid_1
    and voxel_grid_2.  Returns None if the two objects have no
    points in common.
    
    Parameters
    ----------
    voxel_grid_1 : trimesh VoxelGrid object
        First shape to be combined.  Must have same pitch in
        each dimension.
    voxel_grid_2 : trimesh VoxelGrid object
        Second shape to be combined.  Must have same pitch
        in each dimension (and same pitch as voxel_grid_1).

    Returns
    -------
    trimesh VoxelGrid object
        A VoxelGrid with all points from both voxel_grid_1 and voxel_grid_2
    """ 
    
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
    """Displays a protein and its pocket in a Jupyter notebook.
    
    Uses PyVista to show both the protein and the pocket volume.
    The protein and pocket are shown as being hollow; i.e. if
    users zoom past the surface they'll see the inside of the shape.
    The rest of the software package uses filled shapes, but this
    function displays them as hollow to decrease lag.
    
    Parameters
    ----------
    prot_vox : trimesh VoxelGrid object
        The protein
    pocket_vox : trimesh VoxelGrid object
        The pocket volume
    """ 

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