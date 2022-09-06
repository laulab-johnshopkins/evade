import math
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import trimesh
import pyvista as pv
import pandas as pd
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
        The distance that is added to the radius of each atom in the surface.  This is to replicate the
        solvent-accessible surface area.  This value should not be changed after object initialization.
    """
    def __init__(self, mda_atomgroup, solvent_rad=1.09, grid_size=0.7, surf=None, atom_geo_list=None):
        self.mda_atomgroup = mda_atomgroup
        self.solvent_rad = solvent_rad
        
        # Generate surface
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
        self.dict_mda_index_to_atom_geo = {}
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
        all_indices = trimesh.voxel.ops.points_to_indices(voxel_points, pitch=grid_size, origin=[min_x,min_y,min_z])
        self.surf = trimesh.voxel.VoxelGrid(trimesh.voxel.ops.sparse_to_matrix(all_indices))
        
        # The above code makes a surface with the correct shape.  But the surface is too large, and its origin
        # is [0,0,0].  The next few lines fix this.
        self.surf.apply_scale(grid_size)
        self.surf = self.surf.copy() # Necessary due to weird behavior (bug?) in trimesh library.
        self.surf.apply_translation([min(voxel_points[:,0]), min(voxel_points[:,1]), min(voxel_points[:,2])])
        self.surf = self.surf.copy()
        self.surf.fill()
        self.surf = self.surf.copy()



vdw_rads = {"C": 1.7, "H" : 1.2, "N" : 1.55, "O" : 1.52, "S" : 1.8}


def align_to_pocket(protein_surf, pocket_shape, universe,
                    copy_filename, frame_to_align_to):
    """Align an MD trajectory to the coordinates of a pocket.

    Before finding the pocket volume of each frame of the trajectory, it is useful to align the trajectory.
    Global alignment can be too imprecise for this; it's best to align the trajectory to the pocket.  The
    trajectory can be aligned to where the pocket is in any frame; the user must choose which frame.  (The
    first frame is a logical choice.)

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
        Because this function returns an MDAnalysis Universe object, it must create a trajectory file
        for the Universe to read.  The filename is input here.  It needs to be a format that can function
        as a single file; e.g. a multiframe PDB file.  (DCD/PSF doesnt' work because it requires both files.)
    frame_to_align_to : integer
        The frame of the trajectory that other frames should be aligned to.  protein_surf and pocket_shape should
        come from this frame.

    Returns
    -------
    MDAnalysis universe
        An MDAnalysis universe containing the aligned trajectory.
    """

    check_equal_pitches(protein_surf.surf, pocket_shape)
    grid_size = protein_surf.surf.pitch[0]
    solvent_rad = protein_surf.solvent_rad
    # Determine which atoms are in the pocket.
    pocket_atoms_list, pocket_atoms_surf = get_pocket_atoms(protein_surf, pocket_shape,
                                                            universe, solvent_rad,
                                                            grid_size)
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
    print(rms.rmsd(u_copy.atoms.positions, universe.atoms.positions, superposition=False))
    print(rms.rmsd(u_copy.select_atoms(sel_str).positions, universe.select_atoms(sel_str).positions, superposition=False))

    # Align the trajectory.
    u_copy.trajectory[frame_to_align_to]
    mda.analysis.align.AlignTraj(u_copy, universe, select=sel_str, filename=copy_filename).run()
    u_copy=mda.Universe(copy_filename)
    
    # Print RMSDs after alignment.
    universe.trajectory[frame_to_align_to]
    u_copy.trajectory[-1]
    print(rms.rmsd(u_copy.atoms.positions, universe.atoms.positions, superposition=False))
    print(rms.rmsd(u_copy.select_atoms(sel_str).positions, universe.select_atoms(sel_str).positions, superposition=False))
    u_copy.trajectory[frame_to_align_to]

    return u_copy


def correlate_pockets(df_1, df_2):
    """
    Find correlated order parameters between 2 proteins.
    
    Parameters
    ----------
    df_1 : Pandas DataFrame
        This should be the output of compare_frames for one of the pockets of interest.
    df_2 : Pandas DataFrame
        This should be the output of compare_frames for the other pocket of interest.

    Returns
    -------
    Pandas DataFrame
        A DataFrame with information about each pair of order parameters examined by the
        software.
    """
    
    output_df_as_list = []
    for index_1, row_1 in df_1.iterrows():
        for index_2, row_2 in df_2.iterrows():
            pearson, pval = scipy.stats.pearsonr(row_1["Distances"], row_2["Distances"])
            row_1_as_list = row_1.values.tolist()
            row_2_as_list = row_2.values.tolist()
            row_of_output_df = row_1_as_list + row_2_as_list + [pearson, pval]
            output_df_as_list.append(row_of_output_df)
    df = pd.DataFrame(output_df_as_list, columns =["Pocket 1 Atom 1 index",
                                                   "Pocket 1 Atom 2 index",
                                                   "Pocket 1 Pearson of dists",
                                                   "Pocket 1 p-value for Pearson",
                                                   "Pocket 1 Atom 1 name",
                                                   "Pocket 1 Atom 1 residue",
                                                   "Pocket 1 Atom 2 name",
                                                   "Pocket 1 Atom 2 residue",
                                                   "Pocket 1 Rel. change in dist.",
                                                   "Pocket 1 Distances",
                                                   "Pocket 2 Atom 1 index",
                                                   "Pocket 2 Atom 2 index",
                                                   "Pocket 2 Pearson of dists",
                                                   "Pocket 2 p-value for Pearson",
                                                   "Pocket 2 Atom 1 name",
                                                   "Pocket 2 Atom 1 residue",
                                                   "Pocket 2 Atom 2 name",
                                                   "Pocket 2 Atom 2 residue",
                                                   "Pocket 2 Rel. change in dist.",
                                                   "Pocket 2 Distances",
                                                   "Pearson between dists",
                                                   "p-value for pearson between lists"])
    return df


def compare_frames(traj_index_big, traj_index_small, u, protein_surface_big, protein_surface_small, pocket_big, pocket_small, volumes,
                   frames_for_volumes, heavy_atoms=True):
    """
    Get order parameters that quantify the conformational change between two pockets.
    
    Parameters
    ----------
    traj_index_big : int
        The frame number (0-indexed) of the selected frame where the pocket volume
        is bigger.
    traj_index_small : int
        The frame number (0-indexed) of the selected frame where the pocket volume
        is smaller.
    u : MDAnalysis universe
        The universe object that the data are taken from.
    protein_surface_big : ProteinSurface object
        A ProteinSurface for the selected frame where the pocket volume
        is bigger.
    protein_surface_small : ProteinSurface object
        A ProteinSurface for the selected frame where the pocket volume
        is smaller.
    pocket_big : trimesh VoxelGrid object
        The pocket interior of the selected frame where the pocket volume
        is bigger.  This is found by subtracting the protein surface from the
        region of interest
    pocket_small : trimesh VoxelGrid object
        The pocket interior of the selected frame where the pocket volume
        is smaller.
    volumes : list of floats
        The volume of each frame in frames_for_volumes.
    frames_for_volumes : slice of MDAnalysis trajectory
        The frames that prots, pockets, and volumes contain.
        E.g. u.trajectory[0:50]
    heavy_atoms : bool, optional
        When the algorithm checks a hydrogen atoms, and heavy_atoms is True,
        the software will switch to the heavy atom that the H is bound to.  The
        default value of heavy_atoms is True.

    Returns
    -------
    Pandas DataFrame
        A DataFrame with information about each order parameter examined by the
        software.
    """

    check_equal_pitches(protein_surface_big.surf, pocket_big)
    check_equal_pitches(protein_surface_big.surf, pocket_small)
    check_equal_pitches(protein_surface_small.surf, pocket_small)
    grid_size = protein_surface_small.surf.pitch[0]
    if protein_surface_big.solvent_rad == protein_surface_small.solvent_rad:
        solvent_rad = protein_surface_big.solvent_rad
    else:
        raise ValueError("protein_surface_big and protein_surface_small have different solvent_rad values")

    pocket_atoms_frame_big, pocket_frame_big = get_pocket_atoms(protein_surface_big, pocket_big, u, solvent_rad=solvent_rad, grid_size=grid_size)
    pocket_atoms_frame_small, pocket_frame_small = get_pocket_atoms(protein_surface_small, pocket_small, u, solvent_rad=solvent_rad, grid_size=grid_size)
    
    pocket_big_mda_indices = []
    for atom in pocket_atoms_frame_big:
        pocket_big_mda_indices.append(atom.mda_atomgroup[0].index)

    pocket_small_mda_indices = []
    for atom in pocket_atoms_frame_small:
        pocket_small_mda_indices.append(atom.mda_atomgroup[0].index)
    
    # Find out how much each atom moved between the two frames of interest.  This will
    # be used to choose atoms that are unusually mobile or stationary.
    indices_in_either_pocket = list(set(pocket_big_mda_indices + pocket_small_mda_indices))
    dict_index_to_dist = {}
    for atom_index in indices_in_either_pocket:
        u.trajectory[traj_index_big]
        frame_a_pos = u.atoms[atom_index].position
        u.trajectory[traj_index_small]
        frame_b_pos = u.atoms[atom_index].position
        dist = math.dist(frame_a_pos, frame_b_pos)
        dict_index_to_dist[atom_index] = dist
        
    # Find all atoms that moved more/less than most others.
    dist_array = np.array(list(dict_index_to_dist.values()))
    mean_dist = np.mean(dist_array)
    std_dist = np.std(dist_array)
    outlier_indices = []
    print("mean and std dist", mean_dist, std_dist)
    for index, dist in dict_index_to_dist.items():
        if (dist > (mean_dist + std_dist)) or (dist < (mean_dist - 0.5*std_dist)):
            print("outlier", index, dist, u.atoms[index].resid)
            outlier_indices.append(index)
            
    # Among atoms chosen above, find how much exposed surface area each atom has in the smaller pocket.
    pocket_small_edge = get_prot_pocket(protein_surface_small.surf, pocket_small)
    dict_index_to_pocket_voxels = {}
    for index in outlier_indices:
        atom_sphere = protein_surface_small.dict_mda_index_to_atom_geo[index].voxel_sphere
        this_atom_pocket_contribution = voxel_and(pocket_small_edge, atom_sphere)
        if this_atom_pocket_contribution:
            this_atom_num_surface_voxels = this_atom_pocket_contribution.filled_count
            print(index, dict_index_to_dist[index], u.atoms[index].resid, this_atom_num_surface_voxels)
            dict_index_to_pocket_voxels[index] = this_atom_num_surface_voxels
    # Choose atoms with relatively high exposed surface area.
    voxel_count_array = np.array(list(dict_index_to_pocket_voxels.values()))
    mean_voxel_count = np.mean(voxel_count_array)
    std_voxel_count = np.std(voxel_count_array)
    print(mean_voxel_count, std_voxel_count)
    key_moving_indices = []
    key_stationary_indices = []
    for index, voxel_count in dict_index_to_pocket_voxels.items():
        if voxel_count > mean_voxel_count:
            print("high voxel count", index, dict_index_to_dist[index], u.atoms[index].resid,
                  u.atoms[index].name, voxel_count)
            if dict_index_to_dist[index] > mean_dist:
                key_moving_indices.append(index)
            else:
                key_stationary_indices.append(index)
    # Iterate over pairs of atoms chosen above.  For each pair, find the distance
    # between atoms in both frames of interest.  Use this to get the "relative difference"
    # of how much the distance changes between the two frames.
    def get_heavy_index(h_index):
        h_atom = u.atoms[h_index]
        if mda.topology.guessers.guess_atom_element(h_atom.type) != "H":
            return h_index
        heavy_atom = h_atom.bonded_atoms[0]
        return heavy_atom.index
    
    dict_index_pair_to_rel_op_dist = {}
    for moving_index in key_moving_indices:
        if heavy_atoms:
            moving_index = get_heavy_index(moving_index)
        for stationary_index in key_stationary_indices:
            if heavy_atoms:
                stationary_index = get_heavy_index(stationary_index)
                if stationary_index == moving_index:
                    continue
            u.trajectory[traj_index_big]
            op_dist_frame_big = math.dist(u.atoms[moving_index].position,
                                        u.atoms[stationary_index].position)
            u.trajectory[traj_index_small]
            op_dist_frame_small = math.dist(u.atoms[moving_index].position,
                                        u.atoms[stationary_index].position)
            # abs(x-y) / ((x+y)/2) = abs(x-y) / mean_value.
            # See https://en.wikipedia.org/wiki/Relative_change_and_difference
            rel_diff = abs(op_dist_frame_big - op_dist_frame_small) / ((op_dist_frame_big + op_dist_frame_small) / 2)
            dict_index_pair_to_rel_op_dist[(moving_index, stationary_index)] = rel_diff
        for other_moving_index in key_moving_indices:
            if other_moving_index > moving_index: # The > avoids double-counting.
                if heavy_atoms:
                    other_moving_index = get_heavy_index(other_moving_index)
                    if other_moving_index == moving_index:
                        continue
                u.trajectory[traj_index_big]
                op_dist_frame_big = math.dist(u.atoms[moving_index].position,
                                            u.atoms[other_moving_index].position)
                u.trajectory[traj_index_small]
                op_dist_frame_small = math.dist(u.atoms[moving_index].position,
                                            u.atoms[other_moving_index].position)
                # abs(x-y) / ((x+y)/2) = abs(x-y) / mean_value.
                # See https://en.wikipedia.org/wiki/Relative_change_and_difference
                rel_diff = abs(op_dist_frame_big - op_dist_frame_small) / ((op_dist_frame_big + op_dist_frame_small) / 2)
                dict_index_pair_to_rel_op_dist[(moving_index, other_moving_index)] = rel_diff
                
    # Choose distances whose relative difference between the two frames is high.  For each of these
    # distances, calculate the distance for every frame in the trajectory.  Find the correlation between
    # the distance and pocket volume.
    rel_op_dist_array = np.array(list(dict_index_pair_to_rel_op_dist.values()))
    mean_rel_op_dist = np.mean(rel_op_dist_array)
    std_rel_op_dist = np.std(rel_op_dist_array)
    output_df_as_list = []
    for index_pair, rel_op_dist in dict_index_pair_to_rel_op_dist.items():
        index_a = index_pair[0]
        index_b = index_pair[1]
        if rel_op_dist > (mean_rel_op_dist + std_rel_op_dist):
            dist_list = []
            for frame in frames_for_volumes:
                op_dist_this_frame = math.dist(u.atoms[index_a].position,
                                               u.atoms[index_b].position)
                dist_list.append(op_dist_this_frame)
            pearson_r, p_value = scipy.stats.pearsonr(dist_list, volumes)
            
            row_of_df_as_list = [index_a, index_b, pearson_r, p_value,
                                 u.atoms[index_a].name, u.atoms[index_a].resid,
                                 u.atoms[index_b].name, u.atoms[index_b].resid,
                                 rel_op_dist, dist_list]
            output_df_as_list.append(row_of_df_as_list)

    df = pd.DataFrame(output_df_as_list, columns =["Atom 1 index", "Atom 2 index",
                                                   "Pearson of dists",
                                                   "p-value for Pearson",
                                                   "Atom 1 name", "Atom 1 residue",
                                                   "Atom 2 name", "Atom 2 residue",
                                                   "Rel. change in dist.", "Distances"])

    return df


def get_pocket_atoms(protein_surface_obj, pocket_surf, universe, solvent_rad=1.09, grid_size=0.7):
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
    solvent_rad : float, optional
        The protein surface is constructed from the protein atoms'
        van der Waals radii plus the radius of a hypothetical solvent molecule.
        The default value is 1.09 (the van der Waals radius of hydrogen); this was
        chosen because POVME uses this value.
    grid_size : float, optional
        The length (in Angstroms) of each side of a voxel.  The default
        value is 0.7.

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
    based on the input parameters.  The box is filled; i.e.
    internal points are occupied.
    
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
    
    vox_1_and_2_points = npi.intersection(voxel_grid_1.points, voxel_grid_2.points)

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


def compare_prots(protein_surface_1, protein_surface_2, color_1="red", sel_regions_1=None,
                  color_2="blue", sel_regions_2=None):
    """Displays two voxelized proteins in a Jupyter notebook.
    
    Uses PyVista to show two proteins.  They are shown together,
    and each protein is shown separately.
    The protein and pocket are shown as being hollow; i.e. if
    users zoom past the surface they'll see the inside of the shape.
    The rest of the software package uses filled shapes, but this
    function displays them as hollow to decrease lag.
    
    Parameters
    ----------
    protein_surface_1 : ProteinSurface object
        One of the proteins to be displayed.
    protein_surface_2 : ProteinSurface object
        The other protein to be displayed.
    color_1 : string, optional
        The color of the first protein.  The default value is "red".
    sel_regions_1 : dictionary mapping MDAnalysis AtomGroups to strings
        The colors for selected regions of the first protein.
    color_2 : string, optional
        The color of the second protein.  The default value is "blue".
    sel_regions_2 : dictionary mapping MDAnalysis AtomGroups to strings
        The colors for selected regions of the second protein.
    """ 

    dict_sel_1_shape_to_color = {}
    non_sel_1_region = protein_surface_1.surf
    if sel_regions_1:
        for sel_region_mda, sel_color in sel_regions_1.items():
            sel_region = None # initialization
            for selected_atom in sel_region_mda:
                sel_atom_geo = protein_surface_1.dict_mda_index_to_atom_geo[selected_atom.index]
                if sel_region:
                    sel_region = voxel_or(sel_region, sel_atom_geo.voxel_sphere)
                else:
                    sel_region = sel_atom_geo.voxel_sphere
            non_sel_1_region = voxel_subtract(non_sel_1_region, sel_region)
            dict_sel_1_shape_to_color[sel_region] = sel_color
        
    dict_sel_2_shape_to_color = {}
    non_sel_2_region = protein_surface_2.surf
    if sel_regions_2:
        for sel_region_mda, sel_color in sel_regions_2.items():
            sel_region = None # initialization
            for selected_atom in sel_region_mda:
                sel_atom_geo = protein_surface_2.dict_mda_index_to_atom_geo[selected_atom.index]
                if sel_region:
                    sel_region = voxel_or(sel_region, sel_atom_geo.voxel_sphere)
                else:
                    sel_region = sel_atom_geo.voxel_sphere
            non_sel_2_region = voxel_subtract(non_sel_2_region, sel_region)
            dict_sel_2_shape_to_color[sel_region] = sel_color
    
    non_sel_1_vox = non_sel_1_region.copy()
    non_sel_1_vox.hollow()
    non_sel_1_trimesh = non_sel_1_vox.as_boxes()
    non_sel_1_pv = pv.wrap(non_sel_1_trimesh)
    
    non_sel_2_vox = non_sel_2_region.copy()
    non_sel_2_vox.hollow()
    non_sel_2_trimesh = non_sel_2_vox.as_boxes()
    non_sel_2_pv = pv.wrap(non_sel_2_trimesh)
    
    pl = pv.Plotter(shape=(2,2))
    pl.add_mesh(non_sel_1_pv, color=color_1)
    if sel_regions_1:
        for sel_region, color in dict_sel_1_shape_to_color.items():
            sel_vox = sel_region.copy()
            sel_vox.hollow()
            sel_trimesh = sel_vox.as_boxes()
            sel_pv = pv.wrap(sel_trimesh)
            pl.add_mesh(sel_pv, color=color)
    pl.add_mesh(non_sel_2_pv, color=color_2)
    if sel_regions_2:
        for sel_region, color in dict_sel_2_shape_to_color.items():
            sel_vox = sel_region.copy()
            sel_vox.hollow()
            sel_trimesh = sel_vox.as_boxes()
            sel_pv = pv.wrap(sel_trimesh)
            pl.add_mesh(sel_pv, color=color)
    pl.subplot(0,1)
    pl.add_mesh(non_sel_1_pv, color=color_1)
    if sel_regions_1:
        for sel_region, color in dict_sel_1_shape_to_color.items():
            sel_vox = sel_region.copy()
            sel_vox.hollow()
            sel_trimesh = sel_vox.as_boxes()
            sel_pv = pv.wrap(sel_trimesh)
            pl.add_mesh(sel_pv, color=color)
    pl.subplot(1,0)
    pl.add_mesh(non_sel_2_pv, color=color_2)
    if sel_regions_2:
        for sel_region, color in dict_sel_2_shape_to_color.items():
            sel_vox = sel_region.copy()
            sel_vox.hollow()
            sel_trimesh = sel_vox.as_boxes()
            sel_pv = pv.wrap(sel_trimesh)
            pl.add_mesh(sel_pv, color=color)
    pl.link_views()
    pl.show()


def show_one_prot(protein_surface, color="red", sel_regions=None):
    """Displays a voxelized protein in a Jupyter notebook.
    
    Uses PyVista to show a proteins.  It is shown as being hollow; i.e. if
    users zoom past the surface they'll see the inside of the shape.
    The rest of the software package uses filled shapes, but this
    function displays them as hollow to decrease lag.
    
    Parameters
    ----------
    protein_surface : ProteinSurface object
        The protein to be displayed.
    color : string, optional
        The color of regions not described by sel_regions.
        The default value is "red".
    sel_regions : dictionary mapping MDAnalysis AtomGroups to strings
        The colors for selected regions of the protein.
    """ 

    dict_sel_shape_to_color = {}
    non_sel_region = protein_surface.surf
    if sel_regions:
        for sel_region_mda, sel_color in sel_regions.items():
            sel_region = None # initialization
            for selected_atom in sel_region_mda:
                sel_atom_geo = protein_surface.dict_mda_index_to_atom_geo[selected_atom.index]
                if sel_region:
                    sel_region = voxel_or(sel_region, sel_atom_geo.voxel_sphere)
                else:
                    sel_region = sel_atom_geo.voxel_sphere
            non_sel_region = voxel_subtract(non_sel_region, sel_region)
            dict_sel_shape_to_color[sel_region] = sel_color
    
    non_sel_vox = non_sel_region.copy()
    non_sel_vox.hollow()
    non_sel_trimesh = non_sel_vox.as_boxes()
    non_sel_pv = pv.wrap(non_sel_trimesh)
    
    pl = pv.Plotter()
    pl.add_mesh(non_sel_pv, color=color)
    if sel_regions:
        for sel_region, color in dict_sel_shape_to_color.items():
            sel_vox = sel_region.copy()
            sel_vox.hollow()
            sel_trimesh = sel_vox.as_boxes()
            sel_pv = pv.wrap(sel_trimesh)
            pl.add_mesh(sel_pv, color=color)
    pl.show()


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