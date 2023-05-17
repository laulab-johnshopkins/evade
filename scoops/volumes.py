import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import trimesh
import pyvista as pv
import numpy_indexed as npi
import scipy
import scipy.stats

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
    mda_atomgroup : MDAnalysis AtomGroup containing one atom
        An AtonGroup containing the atom that was passed during
        object construction.  mda_atomgroup[0] accesses the atom.
    voxel_sphere : trimesh VoxelGrid
        The sphere that was passed during object construction.
    """
    def __init__(self, mda_atom, voxel_sphere):
        self.mda_atomgroup = mda.AtomGroup([mda_atom])
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
    solvent_rad : float
        The distance that is added to the radius of each atom in the surface.  This is to replicate
        the solvent-accessible surface area.  This value should not be changed after object
        initialization.
    """

    def __init__(self, mda_atomgroup, solvent_rad=1.09, grid_size=0.7, surf=None,
                 atom_geo_list=None):
        self.mda_atomgroup = mda_atomgroup
        self.solvent_rad = solvent_rad

        # Generate surface.  The process is iterative; each time it checks if the minimum point in
        # the new voxel is less (in any dimension) than the previous minimum.  Therefore the first
        # point must be initialized separately.  There's probably a way to implement this in fewer
        # lines of code, but the implementation here works so I kept it.
        dict_radius_to_voxel_sphere = {}
        atom = self.mda_atomgroup[0]
        try:
            element = atom.element
        except mda.NoDataError:
            element = mda.topology.guessers.guess_atom_element(atom.type)
        vdw_rad = vdw_rads[element]
        sphere_rad = vdw_rad + solvent_rad
        next_voxel = generate_voxelized_sphere(sphere_rad,
                                                             atom.position, grid_size)
        num_atoms = len(mda_atomgroup)
        max_atom_rad = max(vdw_rads.values()) + solvent_rad
        max_sphere = generate_voxelized_sphere(max_atom_rad,
                                                             [0,0,0], grid_size)
        # The 1.5 prevents imprecision from causing issues
        max_atom_points = int(max_sphere.points.shape[0] * num_atoms * 1.5)
        voxel_points = np.empty(shape=(max_atom_points,3), dtype=np.float64)
        start_index = 0
        end_index = start_index + len(next_voxel.points)
        num_atoms = 0
        voxel_points[start_index:end_index] = next_voxel.points
        num_atoms += 1
        start_index = end_index
        min_x = next_voxel.origin[0]
        min_y = next_voxel.origin[1]
        min_z = next_voxel.origin[2]
        if surf:
            self.surf = surf
            self.atom_geo_list = atom_geo_list
            self.dict_mda_index_to_atom_geo = {}
            for atom_geo in atom_geo_list:
                self.dict_mda_index_to_atom_geo[atom_geo.mda_atomgroup[0].index] = atom_geo
            return

        self.atom_geo_list = []
        new_atom_geo = AtomGeo(atom, next_voxel)
        self.atom_geo_list.append(new_atom_geo)

        self.dict_mda_index_to_atom_geo = {}
        self.dict_mda_index_to_atom_geo[atom.index] = new_atom_geo

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
            # The round() code shifts the point to the nearest multiple of grid_size.  This is
            # necessary because otherwise the origin could be offset from the expected grid,
            # breaking the boolean code.  The second round() statement in each dimension
            # fixes numerical precision issues from multiplying by grid_size.
            exact_trans = atom.position
            approx_trans_x = round(exact_trans[0] / grid_size) * grid_size
            approx_trans_x = round(approx_trans_x, 5)
            approx_trans_y = round(exact_trans[1] / grid_size) * grid_size
            approx_trans_y = round(approx_trans_y, 5)
            approx_trans_z = round(exact_trans[2] / grid_size) * grid_size
            approx_trans_z = round(approx_trans_z, 5)

            #print("next_voxel before trans", next_voxel.points)
            #print("approx_trans:", approx_trans_x, approx_trans_y, approx_trans_z)

            next_voxel.apply_translation(np.array([approx_trans_x, approx_trans_y,
                                                   approx_trans_z]))
            next_voxel = next_voxel.copy()

            end_index = start_index + next_voxel.filled_count

            voxel_points[start_index:end_index] = next_voxel.points
            start_index = end_index
            num_atoms += 1

            min_x = min(min_x, next_voxel.origin[0])
            min_y = min(min_y, next_voxel.origin[1])
            min_z = min(min_z, next_voxel.origin[2])
            new_atom_geo = AtomGeo(atom, next_voxel)
            self.atom_geo_list.append(new_atom_geo)
            self.dict_mda_index_to_atom_geo[atom.index] = new_atom_geo
        voxel_points = voxel_points[0:end_index]
        all_indices = trimesh.voxel.ops.points_to_indices(voxel_points, pitch=grid_size,
                                                          origin=[min_x,min_y,min_z])
        self.surf = trimesh.voxel.VoxelGrid(trimesh.voxel.ops.sparse_to_matrix(all_indices))

        # The above code makes a surface with the correct shape.  But the surface is too large,
        # and its origin is [0,0,0].  The next few lines fix this.
        self.surf.apply_scale(grid_size)
        self.surf = self.surf.copy() # Necessary due to weird behavior (bug?) in trimesh library.
        self.surf.apply_translation([min(voxel_points[:,0]), min(voxel_points[:,1]),
                                     min(voxel_points[:,2])])
        self.surf = self.surf.copy()
        self.surf.fill()
        self.surf = self.surf.copy()



vdw_rads = {"C": 1.7, "H" : 1.2, "N" : 1.55, "O" : 1.52, "S" : 1.8, "SE" : 1.9, "Se" : 1.9, "ZN" : 1.39, "Zn" : 1.39}


def align_to_pocket(protein_surf, pocket_shape, universe,
                    copy_filename, frame_to_align_to, psf_loc=None, step=None, start=None,
                    stop=None):
    """Align an MD trajectory to the coordinates of a pocket.

    Before finding the pocket volume of each frame of the trajectory, it is useful to align the
    trajectory.  Global alignment can be too imprecise for this; it's best to align the trajectory
    to the pocket.  The trajectory can be aligned to where the pocket is in any frame; the user
    must choose which frame.  (The first frame is a logical choice.)

    Parameters
    ----------
    protein_surf : ProteinSurface object
        The surface at the frame being aligned to.
    pocket_shape : trimesh VoxelGrod
        The interior of the pocket for the frame being aligned to.  This is found by subtracting
        the protein surface from the region of interest using voxel_subtract.
    universe : MDAnalysis universe
        The universe object that the data are taken from.
    copy_filename : string
        Because this function returns an MDAnalysis Universe object, it must create a trajectory
        file for the Universe to read.  The filename is input here.  This can be either a PDB file
        or a DCD file.  If it is a DCD, then `psf_loc` must also be given.
    frame_to_align_to : integer
        The frame of the trajectory that other frames should be aligned to.  protein_surf and
        pocket_shape should come from this frame.
    psf_loc : string, optional
        If `copy_filename` is a DCD file, then MDAnalysis also needs a PSF file to read the data.
        If `copy_filename` is a PDB file, then `psf_loc` should not be provided.  (The default
        value is `None`.)
    step : integer or `None`, optional
        This controls how big of a step to take between frames.  E.g. a value of 1 would lead to
        aligning every frame; a value of 2 would lead to aligning every other frame.  (Skipped
        frames are not included in the output universe.)  The default value of `None` behaves
        identically to a value of 1.
    start : integer or `None`, optional
        This controls which frame to start with.  Frames before `start` are not included in the
        output universe.  The default value of `None` causes the code to start at the first frame.
    stop : integer or `None`, optional
        This controls which frame to end with.  Frames at or after `stop` are not included in the
        output universe.  E.g. if `stop=10` and `start=None`, then the trajectory will include the
        first 10 frames (indices 0-9).  The default value of `None` causes the code to go until the
        end of the trajectory.

    Returns
    -------
    MDAnalysis universe
        An MDAnalysis universe containing the aligned trajectory.
    """

    check_equal_pitches(protein_surf.surf, pocket_shape)
    # Determine which atoms are in the pocket.
    pocket_atoms_list, pocket_atoms_surf = get_pocket_atoms(protein_surf, pocket_shape,
                                                            universe)
    pocket_mda_indices = []
    for atom in pocket_atoms_list:
        pocket_mda_indices.append(atom.mda_atomgroup[0].index)
    indices_string = " ".join(str(index) for index in pocket_mda_indices)
    sel_str = "index %s" %(indices_string)

    # MDAnalysis requires 2 universes for alignment.
    u_copy = universe.copy()

    # Print RMSDs before alignment
    universe.trajectory[frame_to_align_to]
    u_copy.trajectory[-1]

    # Align the trajectory.
    u_copy.trajectory[frame_to_align_to]
    mda.analysis.align.AlignTraj(u_copy, universe, select=sel_str,
                                 filename=copy_filename).run(step=step, start=start, stop=stop)
    if psf_loc:
        u_copy=mda.Universe(psf_loc, copy_filename)
    else:
        u_copy=mda.Universe(copy_filename)

    # Print RMSDs after alignment.
    universe.trajectory[frame_to_align_to]
    u_copy.trajectory[-1]
    u_copy.trajectory[frame_to_align_to]

    return u_copy


def get_largest_shape(voxel_grid):
    """
    Get the largest shape in a trimesh VoxelGrid.

    Voxels kitty-corner from the largest shape are classified as being in a
    separate shape; they are discarded.

    Parameters
    ----------
    voxel_grid : trimesh VoxelGrid object
        The input VoxelGrid.  This object is not modified by this function.
    Returns
    -------
    trimesh VoxelGrid object
        A trimesh VoxelGrid object containing the largest shape in the input
        VoxelGrid.
    """

    if not (voxel_grid.pitch[0] == voxel_grid.pitch[1] == voxel_grid.pitch[2]):
        raise ValueError("Pitches must be equal in all 3 dimensions")
    grid_size = voxel_grid.pitch[0]

    """scipy.ndimage assigns an integer label to each collection of contiguous points.
    It classifies kitty-corner points as being in a separate object (unlike POVME).
    It returns two items:
    * A version of the input matrix where empty elements are assigned 0
      and non-empty elements are assigned their label.
    * The number of separate objects in the matrix.  (This would be 1 if there is
      a single shape surrounded by empty space.)"""
    classified_points = scipy.ndimage.label(voxel_grid.matrix)[0]

    # Find out how many points are in each object.  Note that the first object is
    # label 0; i.e. the empty points.
    feature_labels, feature_sizes = np.unique(classified_points, return_counts=True)

    # Goal: Get the label of the biggest shape.
    # numpy.argmax returns the index of the highest value in an array.  If this value is present
    # multiple times, the function returns the index of the first occurrence.  The first entry in
    # feature_sizes is excluded because it counts empty voxels.  The +1 corrects for the first
    # entry being missing.
    index_of_most_populated_label = np.argmax(feature_sizes[1:])+1

    # Create a VoxelGrid where the points in the largest shape are filled.
    biggest_shape_matrix = np.full(classified_points.shape, False)
    for i in range(classified_points.shape[0]):
        for j in range(classified_points.shape[1]):
            for k in range(classified_points.shape[2]):
                next_point = classified_points[i][j][k]
                if next_point == index_of_most_populated_label:
                    biggest_shape_matrix[i][j][k] = True
    output_surf = trimesh.voxel.VoxelGrid(biggest_shape_matrix)
    output_surf.apply_scale(grid_size)
    output_surf = output_surf.copy() # Necessary due to weird behavior (bug?) in trimesh library.
    output_surf.apply_translation(voxel_grid.translation)
    output_surf = output_surf.copy()
    return output_surf


def voxel_surf_from_numpy(data_loc, grid_size):
    """
    Get a trimesh VoxelGrid from a numpy .npy file.

    Parameters
    ----------
    data_loc : string
        The location of a numpy .npy file.  The file should contain points in a voxelized
        shape.
    grid_size : float
        The spacing between grid points
    Returns
    -------
    trimesh VoxelGrid object
        A trimesh VoxelGrid object containing the points in the data file.
    """

    data = np.load(data_loc)
    min_points_in_data = [min(data[:,0]), min(data[:,1]), min(data[:,2])]
    indices = trimesh.voxel.ops.points_to_indices(data, pitch=grid_size, origin=min_points_in_data)
    surf = trimesh.voxel.VoxelGrid(trimesh.voxel.ops.sparse_to_matrix(indices))
    surf.apply_scale(grid_size)
    surf = surf.copy()
    surf.apply_translation(min_points_in_data)
    surf = surf.copy()
    return surf


def get_pocket_atoms(protein_surface_obj, pocket_surf, universe):
    """
    Get all protein atoms that border the pocket.

    Parameters
    ----------
    protein_surface_obj : ProteinSurface
        The protein molecule.  The object must be
        the ProteinSurface type defined in this software.
    pocket_surf : trimesh VoxelGrod
        The pocket.
    universe : MDAnalysis universe
        The universe object that the data are taken from.

    Returns
    -------
    list of AtomGeo objects
        An AtomGeo for each atom bordering the pocket.
    ProteinSurface object
        A surface containing all atoms bordering the pocket.
    """

    pocket_atoms = []
    for atom_geo in protein_surface_obj.atom_geo_list:
        atom_dilated = dilate_voxel_grid(atom_geo.voxel_sphere)
        atom_pocket_overlap = voxel_and(atom_dilated, pocket_surf)
        if atom_pocket_overlap:
            pocket_atoms.append(atom_geo)
    pocket = pocket_atoms[0].voxel_sphere
    indices = [pocket_atoms[0].mda_atomgroup[0].index]
    atom_geo_list = []
    for atom_geo in pocket_atoms[1:]:
        pocket = voxel_or(pocket, atom_geo.voxel_sphere)
        indices.append(atom_geo.mda_atomgroup[0].index)
        atom_geo_list.append(atom_geo)
    indices_string = " ".join(str(index) for index in indices)
    sel_str = "index %s" %(indices_string)
    mda_atomgroup = universe.select_atoms(sel_str)
    pocket_surf = ProteinSurface(mda_atomgroup, surf=pocket, atom_geo_list=atom_geo_list)
    return pocket_atoms, pocket_surf
    
    
def get_pocket_residues(protein_surface_obj, pocket_surf, universe):
    """
    Get a list of residues that border the pocket.
    
    Get the residue indices of each residue containing
    at least one atom that borders the pocket.  The indices are the
    0-indexed numbers assigned by MDAnalysis; these may differ from
    the numbers given in the trajectory.
    
    Parameters
    ----------
    protein_surface_obj : ProteinSurface
        The protein molecule.  The object must be
        the ProteinSurface type defined in this software.
    pocket_surf : trimesh VoxelGrod
        The pocket.
    universe : MDAnalysis universe
        The universe object that the data are taken from.
        
    Returns
    -------
    pocket_residues : list
        A list of each residue index
    """
    
    pocket_atoms, surf_with_atoms = get_pocket_atoms(protein_surface_obj, pocket_surf, universe)
    pocket_residues = []
    for atom in pocket_atoms:
        this_res = atom.mda_atomgroup[0].resindex
        pocket_residues.append(this_res)
    pocket_residues = list(set(pocket_residues))
    return pocket_residues


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
    if not (voxel_grid.pitch[0] == voxel_grid.pitch[1] == voxel_grid.pitch[2]):
        raise ValueError("Pitches must be equal in all 3 dimensions")
    big_array = np.pad(voxel_grid.matrix, pad_width=((1,0), (1,0), (1,0)), constant_values=False)
    dilated_x_orig = voxel_grid.origin[0] - voxel_grid.scale[0]
    dilated_y_orig = voxel_grid.origin[1] - voxel_grid.scale[1]
    dilated_z_orig = voxel_grid.origin[2] - voxel_grid.scale[2]
    dilated_matrix = scipy.ndimage.binary_dilation(big_array)

    dilated_vg = trimesh.voxel.VoxelGrid(dilated_matrix)
    dilated_vg.apply_scale(voxel_grid.pitch[0])
    dilated_vg.apply_translation([dilated_x_orig, dilated_y_orig, dilated_z_orig])
    dilated_vg = dilated_vg.copy()
    return dilated_vg


def generate_voxelized_sphere(radius, center, grid_size):
    """
    Creates a voxelized sphere.

    The function creates a trimesh VoxelGrid sphere
    based on the input parameters.  The sphere is filled;
    i.e. internal points are occupied.

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

    # These lines get the minimum and maximum positions in each
    # dimension, shifted to the nearest grid point.  (Shifting
    # to the nearest grid point is necessary so that all objects
    # with the same grid_size are aligned with each other.)
    # The statements like x_min = round(x_min, 5) are needed because
    # of float-precision issues.  E.g. 31*0.6=18.599999999999998
    # instead of 18.6 on my computer; this destroys the grid spacing.
    
    x_min = round((center[0]-radius) / grid_size) * grid_size
    x_min = round(x_min, 5)
    x_max = round((center[0]+radius) / grid_size) * grid_size
    x_max = round(x_max, 5)
    y_min = round((center[1]-radius) / grid_size) * grid_size
    y_min = round(y_min, 5)
    y_max = round((center[1]+radius) / grid_size) * grid_size
    y_max = round(y_max, 5)
    z_min = round((center[2]-radius) / grid_size) * grid_size
    z_min = round(z_min, 5)
    z_max = round((center[2]+radius) / grid_size) * grid_size
    z_max = round(z_max, 5)

    # Create a list of all points in the grid.
    # See https://stackoverflow.com/a/12891609
    # The complex numbers cause numpy to go from the min to max values
    # (inclusive of both) with the number of points equal to the complex
    # number.  The round takes care of floating-point imprecision that
    # sometimes causes Python to use the wrong number of grid points.
    X, Y, Z = np.mgrid[x_min:x_max:complex(0, round((x_max-x_min)/grid_size+1, 5)),
                       y_min:y_max:complex(0, round((y_max-y_min)/grid_size+1, 5)),
                       z_min:z_max:complex(0, round((z_max-z_min)/grid_size+1, 5))]

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
                                                      origin=[min(points_in_sphere[:,0]),
                                                              min(points_in_sphere[:,1]),
                                                              min(points_in_sphere[:,2])])
    sphere_voxel = trimesh.voxel.VoxelGrid(trimesh.voxel.ops.sparse_to_matrix(all_indices))
    sphere_voxel.apply_scale(grid_size)
    sphere_voxel = sphere_voxel.copy()
    sphere_voxel.apply_translation([min(points_in_sphere[:,0]), min(points_in_sphere[:,1]),
                                    min(points_in_sphere[:,2])])
    sphere_voxel = sphere_voxel.copy()
    return sphere_voxel


def generate_voxelized_box(lengths, center, grid_size):
    """
    Creates a voxelized box.

    The function creates a trimesh VoxelGrid box
    based on the input parameters.  The box is filled; i.e.
    internal points are occupied.

    Parameters
    ----------
    lengths : list-like object containing three floats
        The box's side lengths in x, y, z directions.
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
    # See the generate_voxelized_sphere comment for why I need the second round line.
    x_min = round((center[0]-0.5*lengths[0]+0.5*grid_size) / grid_size) * grid_size
    x_min = round(x_min, 5)
    x_max = round((center[0]+0.5*lengths[0]-0.5*grid_size) / grid_size) * grid_size
    x_max = round(x_max, 5)
    y_min = round((center[1]-0.5*lengths[1]+0.5*grid_size) / grid_size) * grid_size
    y_min = round(y_min, 5)
    y_max = round((center[1]+0.5*lengths[1]-0.5*grid_size) / grid_size) * grid_size
    y_max = round(y_max, 5)
    z_min = round((center[2]-0.5*lengths[2]+0.5*grid_size) / grid_size) * grid_size
    z_min = round(z_min, 5)
    z_max = round((center[2]+0.5*lengths[2]-0.5*grid_size) / grid_size) * grid_size
    z_max = round(z_max, 5)

    # Create a list of all points in the grid.
    # See https://stackoverflow.com/a/12891609
    # The complex numbers cause numpy to go from the min to max values
    # (inclusive of both) with the number of points equal to the complex
    # number.  The round takes care of floating-point imprecision that
    # sometimes causes Python to use the wrong number of grid points.
    X, Y, Z = np.mgrid[x_min:x_max:complex(0, round((x_max-x_min)/grid_size+1, 5)),
                       y_min:y_max:complex(0, round((y_max-y_min)/grid_size+1, 5)),
                       z_min:z_max:complex(0, round((z_max-z_min)/grid_size+1, 5))]
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
    return True


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
    # Source that float comparison after rounding is OK: https://docs.python.org/3/tutorial/floatingpoint.html
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

    # Source that float comparison after rounding is OK: https://docs.python.org/3/tutorial/floatingpoint.html
    vox_1_and_2_points = npi.intersection(voxel_grid_1.points.round(decimals=5), voxel_grid_2.points.round(decimals=5))

    if len(vox_1_and_2_points) == 0:
        return None
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


def show_in_jupyter(object_1, object_2=None, object_3=None, color_1="red", color_2="blue",
                    color_3="green", sel_regions_1=None, sel_regions_2=None, sel_regions_3=None):

    """Displays 1 to 3 objects (ProteinSurface and/or VoxelGrid) in a Jupyter notebook.

    Uses PyVista to show up to three objects.  They are shown together,
    and each object is shown separately.
    The objects are shown as being hollow; i.e. if
    users zoom past the surface they'll see the inside of the shape.
    The rest of the software package uses filled shapes, but this
    function displays them as hollow to decrease lag.

    Parameters
    ----------
    object_1 : SCOOPS ProteinSurface or trimesh VoxelGrid
        An object to be displayed.
    object_2: SCOOPS ProteinSurface or trimesh VoxelGrid, optional
        Another object to be displayed.  The default is `None`, meaning that no object
        is passed for display.
    object_3: SCOOPS ProteinSurface or trimesh VoxelGrid, optional
        Another object to be displayed.  The default is `None`, meaning that no object
        is passed for display.
    color_1 : string, optional
        The color of the first object (except for any regions specified in sel_regions_1).
        The default value is "red".
    color_2 : string, optional
        The color of the second object (except for any regions specified in sel_regions_2).
        The default value is "blue".
    color_3 : string, optional
        The color of the third object (except for any regions specified in sel_regions_3).
        The default value is "green".
    sel_regions_1 : dictionary mapping MDAnalysis AtomGroups to color-name strings, optional
        The colors for selected regions of object_1.
        E.g. `{u.select_atoms("resid 100") : "yellow", u.select_atoms("resid 20") : "black"}`.
        Can only be used if object_1 is a ProteinSurface.  Large selections slow
        the initial rendering.  The default value is `None`.
    sel_regions_2 : dictionary mapping MDAnalysis AtomGroups to color-name strings, optional
        The colors for selected regions of object_2.  See the `sel_regions_1` description for
        more info.
    sel_regions_3 : dictionary mapping MDAnalysis AtomGroups to color-name strings, optional
        The colors for selected regions of object_3.  See the `sel_regions_1` description for
        more info.

    Returns
    -------
    PyVista Plotter object
        The PyVista plotter object that is shown.  It isn't usually necessary to use this
        object, but it allows settings to be changed.  E.g. plotter_obj.set_background("gray")
        changes the background color.
    """

    # If object_1 is a ProteinSurface, then the user can select regions of it to have different
    # colors.
    if isinstance(object_1, ProteinSurface):
        dict_sel_1_shape_to_color = {}
        non_sel_1_region = object_1.surf
        if sel_regions_1:
            for sel_region_mda, sel_color in sel_regions_1.items():
                sel_region = None # initialization
                for selected_atom in sel_region_mda:
                    sel_atom_geo = object_1.dict_mda_index_to_atom_geo[selected_atom.index]
                    if sel_region:
                        sel_region = voxel_or(sel_region, sel_atom_geo.voxel_sphere)
                    else:
                        sel_region = sel_atom_geo.voxel_sphere
                non_sel_1_region = voxel_subtract(non_sel_1_region, sel_region)
                dict_sel_1_shape_to_color[sel_region] = sel_color
    # Users can't select regions of VoxelGrids.
    elif isinstance(object_1, trimesh.voxel.base.VoxelGrid):
        non_sel_1_region = object_1
        if sel_regions_1:
            raise TypeError("Cannot select regions of VoxelGrid.  sel_regions_1 must be None.")
    # If object_1 isn't a ProteinSurface or VoxelGrid, then the user has given invalid input.
    else:
        raise TypeError("object_1 must be scoops.volumes.ProteinSurface or trimesh.voxel.base.VoxelGrid.")

    # Convert object_1 to PyVista
    non_sel_1_vox = non_sel_1_region.copy()
    non_sel_1_vox.hollow()
    non_sel_1_trimesh = non_sel_1_vox.as_boxes()
    non_sel_1_pv = pv.wrap(non_sel_1_trimesh)
    if sel_regions_1:
        sel_1_pvs_and_colors = [] # list of lists because pv objects are unhashable (can't be dict keys)
        for sel_region, color in dict_sel_1_shape_to_color.items():
            sel_vox = sel_region.copy()
            sel_vox.hollow()
            sel_trimesh = sel_vox.as_boxes()
            sel_pv = pv.wrap(sel_trimesh)
            sel_1_pvs_and_colors.append([sel_pv, color])

    # object_2 and object_3 are handled the same way as object_1, except that they may be None.

    # If object_2 is a ProteinSurface, then the user can select regions of it to have different colors.
    if object_2 and isinstance(object_2, ProteinSurface):
        dict_sel_2_shape_to_color = {}
        non_sel_2_region = object_2.surf
        if sel_regions_2:
            for sel_region_mda, sel_color in sel_regions_2.items():
                sel_region = None # initialization
                for selected_atom in sel_region_mda:
                    sel_atom_geo = object_2.dict_mda_index_to_atom_geo[selected_atom.index]
                    if sel_region:
                        sel_region = voxel_or(sel_region, sel_atom_geo.voxel_sphere)
                    else:
                        sel_region = sel_atom_geo.voxel_sphere
                non_sel_2_region = voxel_subtract(non_sel_2_region, sel_region)
                dict_sel_2_shape_to_color[sel_region] = sel_color
    # Users can't select regions of VoxelGrids.
    elif object_2 and isinstance(object_2, trimesh.voxel.base.VoxelGrid):
        non_sel_2_region = object_2
        if sel_regions_2:
            raise TypeError("Cannot select regions of VoxelGrid.  sel_regions_2 must be None.")
    # If object_2 isn't a ProteinSurface or VoxelGrid, then the user has given invalid input.
    elif object_2:
        raise TypeError("object_2 must be scoops.volumes.ProteinSurface or trimesh.voxel.base.VoxelGrid or None.")

    # Convert object_2 to PyVista
    if object_2:
        non_sel_2_vox = non_sel_2_region.copy()
        non_sel_2_vox.hollow()
        non_sel_2_trimesh = non_sel_2_vox.as_boxes()
        non_sel_2_pv = pv.wrap(non_sel_2_trimesh)
        if sel_regions_2:
            sel_2_pvs_and_colors = []
            for sel_region, color in dict_sel_2_shape_to_color.items():
                sel_vox = sel_region.copy()
                sel_vox.hollow()
                sel_trimesh = sel_vox.as_boxes()
                sel_pv = pv.wrap(sel_trimesh)
                sel_2_pvs_and_colors.append([sel_pv, color])

    # If object_3 is a ProteinSurface, then the user can select regions of it to have
    # different colors.
    if object_3 and isinstance(object_3, ProteinSurface):
        dict_sel_3_shape_to_color = {}
        non_sel_3_region = object_3.surf
        if sel_regions_3:
            for sel_region_mda, sel_color in sel_regions_3.items():
                sel_region = None # initialization
                for selected_atom in sel_region_mda:
                    sel_atom_geo = object_3.dict_mda_index_to_atom_geo[selected_atom.index]
                    if sel_region:
                        sel_region = voxel_or(sel_region, sel_atom_geo.voxel_sphere)
                    else:
                        sel_region = sel_atom_geo.voxel_sphere
                non_sel_3_region = voxel_subtract(non_sel_3_region, sel_region)
                dict_sel_3_shape_to_color[sel_region] = sel_color
    # Users can't select regions of VoxelGrids.
    elif object_3 and isinstance(object_3, trimesh.voxel.base.VoxelGrid):
        non_sel_3_region = object_3
        if sel_regions_3:
            raise TypeError("Cannot select regions of VoxelGrid.  sel_regions_3 must be None.")
    # If object_3 isn't a ProteinSurface or VoxelGrid, then the user has given invalid input.
    elif object_3:
        raise TypeError("object_3 must be scoops.volumes.ProteinSurface or trimesh.voxel.base.VoxelGrid or None.")

    # Convert object_3 to PyVista
    if object_3:
        non_sel_3_vox = non_sel_3_region.copy()
        non_sel_3_vox.hollow()
        non_sel_3_trimesh = non_sel_3_vox.as_boxes()
        non_sel_3_pv = pv.wrap(non_sel_3_trimesh)
        if sel_regions_3:
            sel_3_pvs_and_colors = []
            for sel_region, color in dict_sel_3_shape_to_color.items():
                sel_vox = sel_region.copy()
                sel_vox.hollow()
                sel_trimesh = sel_vox.as_boxes()
                sel_pv = pv.wrap(sel_trimesh)
                sel_3_pvs_and_colors.append([sel_pv, color])

    # Create a window with multiple subplots if >1 object is shown.
    if (object_2 is None) and (object_3 is None):
        pl = pv.Plotter()
    else:
        pl = pv.Plotter(shape=(2,2))

    # Graph the first subplot.
    pl.add_mesh(non_sel_1_pv, color=color_1)
    if sel_regions_1:
        for pv_and_color in sel_1_pvs_and_colors:
            sel_pv = pv_and_color[0]
            color = pv_and_color[1]
            pl.add_mesh(sel_pv, color=color)

    if object_2:
        pl.add_mesh(non_sel_2_pv, color=color_2)
        if sel_regions_2:
            for pv_and_color in sel_2_pvs_and_colors:
                sel_pv = pv_and_color[0]
                color = pv_and_color[1]
                pl.add_mesh(sel_pv, color=color)

    if object_3:
        pl.add_mesh(non_sel_3_pv, color=color_3)
        if sel_regions_3:
            for pv_and_color in sel_3_pvs_and_colors:
                sel_pv = pv_and_color[0]
                color = pv_and_color[1]
                pl.add_mesh(sel_pv, color=color)

    # Graph the second subplot.
    if object_2 or object_3:
        pl.subplot(0,1)
        pl.add_mesh(non_sel_1_pv, color=color_1)
        if sel_regions_1:
            for pv_and_color in sel_1_pvs_and_colors:
                sel_pv = pv_and_color[0]
                color = pv_and_color[1]
                pl.add_mesh(sel_pv, color=color)

    # Graph the third subplot.
    if object_2:
        pl.subplot(1,0)
        pl.add_mesh(non_sel_2_pv, color=color_2)
        if sel_regions_2:
            for pv_and_color in sel_2_pvs_and_colors:
                sel_pv = pv_and_color[0]
                color = pv_and_color[1]
                pl.add_mesh(sel_pv, color=color)

    # Graph the fourth subplot.
    if object_3:
        pl.subplot(1,1)
        pl.add_mesh(non_sel_3_pv, color=color_3)
        if sel_regions_3:
            for pv_and_color in sel_3_pvs_and_colors:
                sel_pv = pv_and_color[0]
                color = pv_and_color[1]
                pl.add_mesh(sel_pv, color=color)

    # Show the shapes.
    if object_2 or object_3:
        pl.link_views()
    
    pl.fly_to(pl.pickable_actors[0].center)
    pl.show()
    return pl
