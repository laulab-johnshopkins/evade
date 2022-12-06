import math
import numpy as np
import pandas as pd
import scipy.stats
import MDAnalysis as mda
import  MDAnalysis.analysis.dihedrals
import seaborn as sns
import matplotlib.pyplot as plt


def get_atom_movements(u, step=None, start=None,
                       stop=None, selection="protein"):
    """
    Quantify atoms' motions from their positions at the first frame of the MD trajectory.

    Iterate over the trajectory.  At each studied frame, find all heavy atoms' distances
    from their positions at the first frame.

    Parameters
    ----------
    u : MDAnalysis universe
        The universe object that the data are taken from.  WARNING: this should be aligned based on
        the residues in `selection`.
    step : integer or `None`, optional
        This controls how big of a step to take between frames.  E.g. a value of 1 would lead to
        studying every frame; a value of 2 would lead to studying every other frame.  The default
        value of `None` behaves identically to a value of 1.
    start : integer or `None`, optional
        This controls which frame to start with.  The default value of `None` causes the code to
        start at the first frame.
    stop : integer or `None`, optional
        This controls which frame to end with.  Frames at or after `stop` are ignored.  E.g. if
        `stop=10` and `start=None`, then the code will analyze the first 10 frames (indices 0-9).
        The default value of `None` causes the code to go until the end of the trajectory.
    selection : string, optional
        An MDAnalysis selection string describing the residues to be studied.  The default value of
        "protein" selects all protein residues.  Any hydrogens in `selection` will be ignored.

    Returns
    -------
    dict_index_to_dist_list : dictionary mapping integers to 1D arrays
        Each key is the MDAnalysis atom index of a heavy atom.  (Every heavy atom described by
        `selection` will have a key.)  Each value is an array listing the atom's distance from
        its first-frame position at each studied frame.
    """

    u.trajectory[0]
    u_that_iterates = u.copy()
    u_that_iterates.trajectory[0]
    sel_in_u = u.select_atoms(selection)
    sel_in_u_that_iterates = u_that_iterates.select_atoms(selection)
    
    if not start:
        start = 0
    if not stop:
        stop = len(u.trajectory)
    if not step:
        step = 1

    # Initialize a dictionary mapping MDAnalysis atom indices to arrays listing the atoms'
    # distances from their positions at the first frame.  The arrays are initialized
    # to 0 and modified later in the function.
    dict_index_to_dist_list = {}
    for i in range(len(sel_in_u_that_iterates)):
        moved_atom = sel_in_u_that_iterates[i]
        if mda.topology.guessers.guess_atom_element(moved_atom.type) == "H":
            continue
        empty_dist_list = np.zeros(len(u.trajectory[start:stop:step]))
        dict_index_to_dist_list[moved_atom.ix] = empty_dist_list

    # Get each atom's distance from initial position at each iterated frame.
    frame_num = 0
    for frame in u_that_iterates.trajectory[start:stop:step]:
        for i in range(len(sel_in_u)):
            moved_atom = sel_in_u_that_iterates[i]
            if mda.topology.guessers.guess_atom_element(moved_atom.type) == "H":
                continue
            orig_atom = sel_in_u[i]
            this_dist = math.dist(moved_atom.position,
                                    orig_atom.position)
            dict_index_to_dist_list[moved_atom.ix][frame_num] = this_dist
        frame_num += 1
    return dict_index_to_dist_list


def correlate_dists_to_observable(dict_index_to_dist_list, observable):
    """
    Get the correlations between each residue's distance from first frame position and another
    observable.

    This function should be run after `get_atom_movements`.

    Parameters
    ----------
    dict_index_to_dist_list : dictionary mapping integers to 1D arrays
        This argument should be the output of `get_atom_movements`.  Each key is an MDAnalysis
        atom index; each value is a list of the atom's distance from first frame position at every
        studied frame.  Note that `get_atom_movements` can iterate over frames with step size >= 1.
    observable : list-like
        This contains an observable's value at each trajectory frame that `dict_index_to_dist_list`
        has distances for.

    Returns
    -------
    sorted_indices : list of integers
        A list of keys of `dict_index_to_dist_list`, sorted lowest to highest.
    corrs : list of floats
        The Pearson correlation coefficient between each atom's distance from first frame and
        `observable`.
    pvals : list of floats
        The p-value for each atom's Pearson coefficient.
    """

    sorted_indices = sorted(list(dict_index_to_dist_list.keys()))
    corrs = []
    pvals = []
    for i in sorted_indices:
        corr_motion_with_obs = scipy.stats.pearsonr(dict_index_to_dist_list[i], observable)
        corrs.append(corr_motion_with_obs[0])
        pvals.append(corr_motion_with_obs[1])
    return sorted_indices, corrs, pvals


def get_atoms_that_corr(corr_cutoff, sorted_indices, corrs, u):
    """
    Get atoms whose correlation with an observable has magnitude greater than a cutoff.

    This function should be run after `correlate_dists_to_observable`.  If a residue
    has multiple atoms with correlations beyond the threshold, this function only chooses
    the atom with the strongest correlation.  When comparing correlation values, the function uses
    the absolute value of the Pearson coefficient; therefore the function considers values close to
    either 1 or -1 as being strong.

    Parameters
    ----------
    corr_cutoff : float
        The cutoff for considering correlations to be meaningful.
    sorted_indices : list-like
        A list of MDAnalysis atom indices for all atoms to be studied.  This is returned by
        `correlate_dists_to_observable`.
    corrs : list of floats
        The Pearson correlation coefficient between each atom's distance from first frame and an
        observable.  This is returned by `correlate_dists_to_observable`.
    u : MDAnalysis universe
        The universe object that the data are taken from.  It is likely the same object used
        as an argument to `get_atom_movements`.

    Returns
    -------
    atom_indices_that_corr : list of integers
        A list of MDAnalysis atom indices of atoms whose motion correlates with an observable.
    """

    corr_cutoff = abs(corr_cutoff) # in case the user inputs a negative value

    atom_indices_that_corr = []
    last_atom_resnum = None # initialization
    last_atom_corr = None
    for i in range(len(sorted_indices)):
        if abs(corrs[i]) > corr_cutoff:
            atom_index = sorted_indices[i]
            # If the code finds multiple atoms from same residue, only take the atom with
            # strongest correlation.
            if u.atoms[atom_index].resnum == last_atom_resnum:
                if abs(last_atom_corr) < abs(corrs[i]):
                    atom_indices_that_corr[-1] = atom_index
                    last_atom_corr = corrs[i]
            else:
                atom_indices_that_corr.append(atom_index)
                last_atom_corr = corrs[i]
                last_atom_resnum = u.atoms[atom_index].resnum
    return atom_indices_that_corr


def explain_atom_corr_with_observable(atom, observable, observable_label, u,
                                      dict_index_to_dist_list,
                                      dihedral_pearson_cutoff=0.3, step=None, start=None,
                                      stop=None):
    """
    Determine if an atom's correlation with an observable can be explained by a dihedral angle.

    This function should be run after `get_atoms_that_corr`.  It operates on a single atom,
    checking if a dihedral on that atom's residue correlates with the observable.  The function
    considers both positive and negative correlations to be meaningful.

    Parameters
    ----------
    atom : MDAnalysis Atom object
        An atom whose distance from first-frame position correlates with `obervable`.
    observable : list-like
        This contains an observable's value at each analyzed frame of an MD trajectory.
    observable_label : string
        A short description of the observable, to be used when labeling graphs created by the
        function.
    u : MDAnalysis universe
        The universe object that the data are taken from.  It is likely the same object used
        as an argument to `get_atom_movements`.
    dict_index_to_dist_list : dictionary mapping integers to 1D arrays
        This argument should be the output of `get_atom_movements`.  Each key is an MDAnalysis atom
        index; each value is a list of the atom's distance from first frame position at every
        studied frame.
    dihedral_pearson_cutoff : float, optional
        The minimum correlation strength between a dihedral and `observable` that is considered
        meaningful.  The default value is 0.3.  The function uses the absolute value of
        correlations to include negative correlations; thus `dihedral_pearson_cutoff` should always
        be positive.
    step : integer or `None`, optional
        This controls how big of a step to take between frames.  E.g. a value of 1 would lead to
        studying every frame; a value of 2 would lead to studying every other frame.  The default
        value of `None` behaves identically to a value of 1.
    start : integer or `None`, optional
        This controls which frame to start with.  The default value of `None` causes the code to
        start at the first frame.
    stop : integer or `None`, optional
        This controls which frame to end with.  Frames at or after `stop` are ignored.  E.g. if
        `stop=10` and `start=None`, then the code will analyze the first 10 frames (indices 0-9).
        The default value of `None` causes the code to go until the end of the trajectory.

    Returns
    -------
    dihedral_dict : dictionary
        A dictionary listing various results about the dihedrals.  It contains:

        ``"distance_from_first_frame"``
            1D array of floats containing the atom's distance from first-frame position at each
            studied frame.
        ``"distance_from_first_frame_pearson"``
            A scipy PearsonRResult object correlating the phi dihedral with `observable`.
            `dihedral_dict["distance_from_first_frame_pearson"][0]` is the Pearson coefficient,
            while `dihedral_dict["distance_from_first_frame_pearson"][1]` is the p-value.
        ``"phi"``
            Phi dihedral for `atom`'s residue at each studied frame.  Is `None` if not defined for
            this residue (e.g. because of its position at end of chain).
        ``"phi_pearson"``
            A scipy PearsonRResult object correlating the phi dihedral with `observable`.
            `dihedral_dict["phi_pearson"][0]` is the Pearson coefficient, while
            `dihedral_dict["phi_pearson"][1]` is the p-value.
        ``"psi"``
            Phi dihedral for `atom`'s residue at each studied frame.  Is `None` if not defined for
            this residue (e.g. because of its position at end of chain).
        ``"psi_pearson"``
            A scipy PearsonRResult object correlating the psi dihedral with `observable`.
        ``"chi1"``
            Chi1 dihedral for `atom`'s residue at each studied frame.  Is `None` if not defined for
            this residue (e.g. because sidechain is too short).
        ``"chi1_pearson"``
            A scipy PearsonRResult object correlating the chi1 dihedral with `observable`.
        ``"chi2"``
            Chi2 dihedral for `atom`'s residue at each studied frame.  Is `None` if not defined for
            this residue (e.g. because sidechain is too short).
        ``"chi2_pearson"``
            A scipy PearsonRResult object correlating the chi2 dihedral with `observable`.
    """

    # in case the user inputs a negative value
    dihedral_pearson_cutoff = abs(dihedral_pearson_cutoff)

    atom_label = "Atom %s of residue %s %d of chain %s" %(atom.name,
                                                          atom.resname,
                                                          atom.resid,
                                                          atom.segid)

    residue_label = "%s %s" %(atom.segid, atom.resid)
    print("0-indexed atom index:", atom.ix)
    print(atom_label)

    distance_from_first_frame_pearson = scipy.stats.pearsonr(dict_index_to_dist_list[atom.ix],
    observable)
    print(distance_from_first_frame_pearson)
    sns.kdeplot(x=dict_index_to_dist_list[atom.ix], y=observable,
                fill=True, cmap="gnuplot2_r")
    sns.kdeplot(x=dict_index_to_dist_list[atom.ix], y=observable, color="black")
    plt.xlabel("Atom's Distance From Position In First MD Frame")
    plt.ylabel(observable_label)
    plt.show()

    # This creates a residuegroup with 1 residue.  I don't remember why this was preferable
    # to directly storing the residue, but I think there was a reason.
    resindex = atom.resindex
    residuegroup = u.residues[resindex : resindex+1]

    phi_sel = [res.phi_selection() for res in residuegroup]
    if phi_sel[0]:
        phi_angles_obj = MDAnalysis.analysis.dihedrals.Dihedral(phi_sel).run(step=step, start=start, stop=stop)
        phi_angles = phi_angles_obj.results.angles.flatten()
        phi_pearson = scipy.stats.pearsonr(phi_angles, observable)
        if abs(phi_pearson[0]) > dihedral_pearson_cutoff:
            print(phi_pearson)
            sns.kdeplot(x=phi_angles, y=observable,
                    fill=True, cmap="gnuplot2_r")
            sns.kdeplot(x=phi_angles, y=observable, color="black")
            plt.xlabel("%s Phi" %(residue_label))
            plt.ylabel(observable_label)
            plt.show()
    else:
        print("WARNING: residue doesn't have phi dihedral.  It might be at the end of a chain.")
        phi_angles = None
        phi_pearson = None


    psi_sel = [res.psi_selection() for res in residuegroup]
    if psi_sel[0]:
        psi_angles_obj = MDAnalysis.analysis.dihedrals.Dihedral(psi_sel).run(step=step, start=start, stop=stop)
        psi_angles = psi_angles_obj.results.angles.flatten()
        psi_pearson = scipy.stats.pearsonr(psi_angles, observable)
        if abs(psi_pearson[0]) > dihedral_pearson_cutoff:
            print(psi_pearson)
            sns.kdeplot(x=psi_angles, y=observable,
                    fill=True, cmap="gnuplot2_r")
            sns.kdeplot(x=psi_angles, y=observable, color="black")
            plt.xlabel("%s Psi" %(residue_label))
            plt.ylabel(observable_label)
            plt.show()
    else:
        print("WARNING: residue doesn't have psi dihedral.  It might be at the end of a chain.")
        psi_angles = None
        psi_pearson = None

    if atom.resname in ["ALA", "GLY"]:
        chi1_angles = None
        chi1_pearson = None
        chi2_angles = None
        chi2_pearson = None

    elif atom.resname in ["CYS", "SER", "THR", "VAL", "PRO"]:
        chi1_sel = [res.chi1_selection() for res in residuegroup]
        chi1_angles_object = MDAnalysis.analysis.dihedrals.Dihedral(chi1_sel).run(step=step, start=start, stop=stop)
        chi1_angles = chi1_angles_object.results.angles.flatten()

        chi1_pearson = scipy.stats.pearsonr(chi1_angles, observable)
        if abs(chi1_pearson[0]) > dihedral_pearson_cutoff:
            print(chi1_pearson)
            sns.kdeplot(x=chi1_angles, y=observable,
                    fill=True, cmap="gnuplot2_r")
            sns.kdeplot(x=chi1_angles, y=observable, color="black")
            plt.xlabel("%s Chi1" %(residue_label))
            plt.ylabel(observable_label)
            plt.show()
        chi2_angles = None
        chi2_pearson = None

    else:
        janin_results = MDAnalysis.analysis.dihedrals.Janin(u.residues[resindex].atoms).run(step=step, start=start, stop=stop)

        chi1_angles = janin_results.results.angles[:,:,0][::].flatten()
        chi1_pearson = scipy.stats.pearsonr(chi1_angles, observable)
        if abs(chi1_pearson[0]) > dihedral_pearson_cutoff:
            print(chi1_pearson)
            sns.kdeplot(x=chi1_angles, y=observable,
                    fill=True, cmap="gnuplot2_r")
            sns.kdeplot(x=chi1_angles, y=observable, color="black")
            plt.xlabel("%s Chi1" %(residue_label))
            plt.ylabel(observable_label)
            plt.show()

        chi2_angles = janin_results.results.angles[:,:,1][::].flatten()
        chi2_pearson = scipy.stats.pearsonr(chi2_angles, observable)
        if abs(chi2_pearson[0]) > dihedral_pearson_cutoff:
            print(chi2_pearson)
            sns.kdeplot(x=chi2_angles, y=observable,
                    fill=True, cmap="gnuplot2_r")
            sns.kdeplot(x=chi2_angles, y=observable, color="black")
            plt.xlabel("%s Chi2" %(residue_label))
            plt.ylabel(observable_label)
            plt.show()

    dihedral_dict = {"distance_from_first_frame":dict_index_to_dist_list[atom.ix],
                     "distance_from_first_frame_pearson":distance_from_first_frame_pearson,
                     "phi":phi_angles, "phi_pearson":phi_pearson,
                     "psi":psi_angles, "psi_pearson":psi_pearson,
                     "chi1":chi1_angles, "chi1_pearson":chi1_pearson,
                     "chi2":chi2_angles, "chi2_pearson":chi2_pearson}
    return dihedral_dict


def get_best_dihedrals(dict_atom_index_to_dihedrals, min_pearson_to_include=0.0):
    """
    Find the dihedral of each residue of interest that correlates most with an observable.

    Given a set of dihedrals for each atom of interest and their correlations with an observable,
    extract the dihedral of each angle whose correlation (positive or negative) is the strongest.

    Parameters
    ----------
    dict_atom_index_to_dihedrals : dictionary
        Each key is a 0-indexed MDAnalysis atom index.  Each value is the output of
        `explain_atom_corr_with_observable`.
    min_pearson_to_include : float, optional
        The minimum Pearson between the dihedral and observable needed for the angle to be
        included.  If a residue has no dihedrals whose Pearson exceeds this threshold, then the
        residue will not be included in the output.  This function takes the absolute value of
        Pearsons to consider both positive and negative correlations as valid; thus
        `min_pearson_to_include` should be positive.  The default value is 0, meaning that the best
        dihedral for a residue will be returned even when that dihedral has negligible correlation.

    Returns
    -------
    df_best_angles : Pandas DataFrame
        Columns are of the form `index_angle`, e.g. `64_chi1` for the chi1 dihedral of 0-indexed
        atom 64.  Rows are the dihedral value at each studied time point.
    """

    min_pearson_to_include = abs(min_pearson_to_include) # in case user gives a negative value.
    dict_best_angle_label_to_values = {}
    for atom_index, data_dict in dict_atom_index_to_dihedrals.items():
        min_pearson = min_pearson_to_include
        max_angle = None
        for dihedral in ["phi", "psi", "chi1", "chi2"]:
            this_pearson_obj = dict_atom_index_to_dihedrals[atom_index]["%s_pearson" %(dihedral)]
            if this_pearson_obj is None:
                continue
            this_pearson = this_pearson_obj[0]
            if abs(this_pearson) > min_pearson:
                max_angle = dict_atom_index_to_dihedrals[atom_index][dihedral]
                max_angle_label = "%d_%s" %(atom_index, dihedral)
                min_pearson = abs(this_pearson)
        if max_angle is not None:
            dict_best_angle_label_to_values[max_angle_label] = max_angle

    df_best_angles = pd.DataFrame(dict_best_angle_label_to_values)
    return df_best_angles


def add_correlated_dihedrals(nglview_widget, u, corr_mat, corr_cutoff):
    """
    Show correlated dihedrals in an NGLView display.

    Parameters
    ----------
    nglview_widget : NGLView Widget
        An NGLView widget object.
    u : MDAnalysis Universe
        An MDAnalysis Universe containing the system shown in `nglview_widget`.
    corr_mat : Pandas DataFrame
        The correlation matrix produced by dataframe.corr().
    corr_cutoff : float
        The minimum correlation strength needed for the correlation to be displayed.  The function
        takes the absolute value of the correlations to consider both positive and negative
        correlations as significant; thus `corr_cutoff` should be positive.

    Notes
    -----
    The function modifies the input object `nglview_widget`, as described below:

    A sphere is drawn at each key atom's position.  A line is drawn between each shown correlation.
    The line thickness is determined by correlation strength.  Hovering the mouse over a line
    causes it to display info about the correlation.
    """

    corr_cutoff = abs(corr_cutoff)
    for corr_item in corr_mat:
        atom_index = int(corr_item.split("_")[0])
        atom = u.atoms[atom_index]
        nglview_widget.shape.add_sphere(atom.position, [0,1,1], 1.0)
        atom_resid = atom.resid
        atom_chainid = atom.chainID
        nglview_widget.add_ball_and_stick("%d:%s" %(atom_resid, atom_chainid))

    angle_labels = corr_mat.columns
    for i in range(len(angle_labels)):
        for j in range(i+1, len(angle_labels)):
            if abs(corr_mat.iloc[i,j]) > corr_cutoff:
                atom_i_index = int(angle_labels[i].split("_")[0])
                atom_j_index = int(angle_labels[j].split("_")[0])

                cylinder_name = ("corr between chain %s resid %s (%s) and "
                                   "chain %s resid %s (%s) is %f" %(u.atoms[atom_i_index].chainID,
                                                                   u.atoms[atom_i_index].resid, angle_labels[i],
                                                                   u.atoms[atom_j_index].chainID,
                                                                   u.atoms[atom_j_index].resid, angle_labels[j],
                                                                   corr_mat.iloc[i,j]))
                nglview_widget.shape.add_cylinder(u.atoms[atom_i_index].position,
                                                  u.atoms[atom_j_index].position, [0,1,1],
                                                  abs(corr_mat.iloc[i,j]), cylinder_name)
