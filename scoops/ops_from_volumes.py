import math
import scipy.stats
import pandas as pd
import numpy as np
import MDAnalysis as mda
from . import volumes

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
                                                   "Pocket 1 Atom 1 chain or segid",
                                                   "Pocket 1 Atom 2 name",
                                                   "Pocket 1 Atom 2 residue",
                                                   "Pocket 1 Atom 2 chain or segid",
                                                   "Pocket 1 Rel. change in dist.",
                                                   "Pocket 1 Distances",
                                                   "Pocket 2 Atom 1 index",
                                                   "Pocket 2 Atom 2 index",
                                                   "Pocket 2 Pearson of dists",
                                                   "Pocket 2 p-value for Pearson",
                                                   "Pocket 2 Atom 1 name",
                                                   "Pocket 2 Atom 1 residue",
                                                   "Pocket 2 Atom 1 chain or segid",
                                                   "Pocket 2 Atom 2 name",
                                                   "Pocket 2 Atom 2 residue",
                                                   "Pocket 2 Atom 2 chain or segid",
                                                   "Pocket 2 Rel. change in dist.",
                                                   "Pocket 2 Distances",
                                                   "Pearson between dists",
                                                   "p-value for pearson between lists"])
    return df


def compare_frames(traj_index_big, traj_index_small, u, protein_surface_big, protein_surface_small,
                   pocket_big, pocket_small, vols_list, frames_for_volumes, heavy_atoms=True,
                   chain_or_segid="segid", verbose=False):
    """
    Get order parameters that quantify the conformational change between two pockets.

    Each order parameter is a distance between two atoms.

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
    vols_list : list of floats
        The volume of each frame in frames_for_volumes.
    frames_for_volumes : slice of MDAnalysis trajectory
        The frames that prots, pockets, and volumes contain.
        E.g. u.trajectory[0:50]
    heavy_atoms : bool, optional
        When the algorithm checks a hydrogen atoms, and heavy_atoms is True,
        the software will switch to the heavy atom that the H is bound to.  The
        default value of heavy_atoms is True.
    chain_or_segid : string or None, optional
        Whether to list the chain, segid, or neither.  Acceptable values are `"chain"`, `"segid"`,
        or `None`.  The default value is `"segid"`.
    verbose : bool, optional
        Whether to print the results of calculations performed during the analysis.
        The default value is `False`.

    Returns
    -------
    Pandas DataFrame
        A DataFrame with information about each order parameter examined by the
        software.
    Returns
    -------
    Pandas DataFrame
        A DataFrame with information about each order parameter examined by the
        software.  There are quite a few columns; the most interesting are:

        ``"Atom 1 index"``
            0-indexed atom index of an atom in the pair.
        ``"Atom 2 index"``
            0-indexed atom index of the other atom in the pair.
        ``"Pearson of dists"``
            Pearson coefficient between the atom-atom distance and the pocket volume
            over all frames in `frames_for_volumes`.
        ``"p-value for Pearson"``
            p-value for the Pearson coefficient.
        ``"Rel. change in dist."``
            The change in atom-atom distance between the two frames of interest, processed
            based on distance magnitude according to
            https://en.wikipedia.org/wiki/Relative_change_and_difference.
        ``"Distances"``
            The atom-atom distance at each studied frame.
    """

    volumes.check_equal_pitches(protein_surface_big.surf, pocket_big)
    volumes.check_equal_pitches(protein_surface_big.surf, pocket_small)
    volumes.check_equal_pitches(protein_surface_small.surf, pocket_small)
    if protein_surface_big.solvent_rad != protein_surface_small.solvent_rad:
        raise ValueError("protein_surface_big and protein_surface_small have different "
                         "solvent_rad values")

    pocket_atoms_frame_big, pocket_frame_big = volumes.get_pocket_atoms(protein_surface_big,
                                                                        pocket_big, u)
    pocket_atoms_frame_small, pocket_frame_small = volumes.get_pocket_atoms(protein_surface_small,
                                                                            pocket_small, u)

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
    if verbose:
        print("Mean distance moved by pocket atoms between two selected frames:", mean_dist)
        print("standard deviation of this distance:", std_dist)
        print("Listing exceptionally mobile/still atoms:")
    for index, dist in dict_index_to_dist.items():
        if (dist > (mean_dist + std_dist)) or (dist < (mean_dist - 0.5*std_dist)):
            if verbose:
                print("atom index:", index, "movement between frames:", dist,
                    "residue:", u.atoms[index].resid)
            outlier_indices.append(index)

    # Among atoms chosen above, find how much exposed surface area each atom has in the
    # smaller pocket.
    if verbose:
        print("Finding atoms from above list that have voxels on the pocket surface.")
    pocket_small_edge = volumes.get_prot_pocket(protein_surface_small.surf, pocket_small)
    dict_index_to_pocket_voxels = {}
    for index in outlier_indices:
        atom_sphere = protein_surface_small.dict_mda_index_to_atom_geo[index].voxel_sphere
        this_atom_pocket_contribution = volumes.voxel_and(pocket_small_edge, atom_sphere)
        if this_atom_pocket_contribution:
            this_atom_num_surface_voxels = this_atom_pocket_contribution.filled_count
            dict_index_to_pocket_voxels[index] = this_atom_num_surface_voxels
            if verbose:
                print("atom index:", index, "movement between frames:", dict_index_to_dist[index],
                      "residue:", u.atoms[index].resid,
                      "atom's voxels on pocket surface:", this_atom_num_surface_voxels)
    # Choose atoms with relatively high exposed surface area.
    voxel_count_array = np.array(list(dict_index_to_pocket_voxels.values()))
    mean_voxel_count = np.mean(voxel_count_array)
    std_voxel_count = np.std(voxel_count_array)
    if verbose:
        print("mean pocket-exposed voxels (from above atoms):", mean_voxel_count, "std",
              std_voxel_count)
    key_moving_indices = []
    key_stationary_indices = []
    if verbose:
        print("Finding atoms from above list with higher-than-average pocket surface exposure.")
    for index, voxel_count in dict_index_to_pocket_voxels.items():
        if voxel_count > mean_voxel_count:
            if verbose:
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
    # distances, calculate the distance for every frame in the trajectory.  Find the correlation
    # between the distance and pocket volume.
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
            pearson_r, p_value = scipy.stats.pearsonr(dist_list, vols_list)

            if chain_or_segid == "chain":
                a_chain_or_segid = u.atoms[index_a].chainID
                b_chain_or_segid = u.atoms[index_b].chainID
            elif chain_or_segid == "segid":
                a_chain_or_segid = u.atoms[index_a].segid
                b_chain_or_segid = u.atoms[index_b].segid
            else:
                a_chain_or_segid = None
                b_chain_or_segid = None

            row_of_df_as_list = [index_a, index_b, pearson_r, p_value,
                                 u.atoms[index_a].name, u.atoms[index_a].resid, a_chain_or_segid,
                                 u.atoms[index_b].name, u.atoms[index_b].resid, b_chain_or_segid,
                                 rel_op_dist, dist_list]
            output_df_as_list.append(row_of_df_as_list)

    df = pd.DataFrame(output_df_as_list, columns =["Atom 1 index", "Atom 2 index",
                                                   "Pearson of dists",
                                                   "p-value for Pearson",
                                                   "Atom 1 name", "Atom 1 residue",
                                                   "Atom 1 chain or segid",
                                                   "Atom 2 name", "Atom 2 residue",
                                                   "Atom 2 chain or segid",
                                                   "Rel. change in dist.", "Distances"])

    return df
