import numpy as np
import scipy.stats
import MDAnalysis as mda
import MDAnalysis.analysis.dihedrals
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.feature_selection


def get_dihedrals_for_resindex_list(resindex_list, u, step=None, start=None, stop=None):
    """
    Get the dihedrals for a residue across an MD trajectory.

    This function gets the phi, psi, chi1, and chi2 dihedrals for a residue.
    Any undefined angles are set to `None`.  In most cases the chi1 and chi2
    angles are shifted from [-180, 180] to [0, 360] using the equation
    `(angle + 360) % 360`.  But for CYS, SER, THR, VAL, and PRO, chi2 is
    undefined and chi1 is in [-180, 180].

    Parameters
    ----------
    resindex_list : list (or 1D array) of integers
        The 0-indexed residue index assigned by MDAnalysis to each residue of
        interest.
    u : MDAnalysis universe
        The universe object that the data are taken from.
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
    dihed_dict : nested dictionary
        This stores the values across the MD trajectory of each residue's
        dihedrals.  Dihedrals are accessed using
        `dihed_dict[resindex][dihedral]`, e.g. `dihed_dict[3]["psi"]`.
        Available dihedrals are "phi", "psi", "chi1", and "chi2".  Each
        dihedral is a 1D array of angles, or `None` for undefined dihedrals.
    """

    dihed_dict = {}
    for resindex in resindex_list:
        residuegroup = u.residues[resindex : resindex+1]

        phi_sel = [res.phi_selection() for res in residuegroup]
        if phi_sel[0]:
            phi_angles_obj = mda.analysis.dihedrals.Dihedral(phi_sel).run(step=step, start=start,
                                                                          stop=stop)
            phi_angles = phi_angles_obj.results.angles.flatten()
        else:
            print("WARNING: residue doesn't have phi dihedral.  It might be at the end of a chain.")
            phi_angles = None

        psi_sel = [res.psi_selection() for res in residuegroup]
        if psi_sel[0]:
            psi_angles_obj = mda.analysis.dihedrals.Dihedral(psi_sel).run(step=step, start=start,
                                                                          stop=stop)
            psi_angles = psi_angles_obj.results.angles.flatten()
        else:
            print("WARNING: residue doesn't have psi dihedral.  It might be at the end of a chain.")
            psi_angles = None

        if u.residues[resindex].resname in ["ALA", "GLY"]:
            chi1_angles = None
            chi2_angles = None

        elif u.residues[resindex].resname in ["CYS", "SER", "THR", "VAL", "PRO"]:
            chi1_sel = [res.chi1_selection() for res in residuegroup]
            chi1_angles_obj = mda.analysis.dihedrals.Dihedral(chi1_sel).run(step=step, start=start,
                                                                            stop=stop)
            chi1_angles = chi1_angles_obj.results.angles.flatten()
            chi2_angles = None

        else:
            janin_obj = mda.analysis.dihedrals.Janin(u.residues[resindex].atoms)
            janin_results = janin_obj.run(step=step, start=start, stop=stop)
            chi1_angles = janin_results.results.angles[:,:,0][::].flatten()
            chi2_angles = janin_results.results.angles[:,:,1][::].flatten()

        dihed_dict[resindex] = {"phi":phi_angles, "psi":psi_angles, "chi1":chi1_angles,
                                   "chi2":chi2_angles}
    return dihed_dict


def get_key_pocket_dihedrals(pocket_resindex_list, all_resindex_list, dict_all_res_diheds, u,
                             neighbors_to_exclude, score_cutoff, score):
    """
    Find pocket residues whose dihedrals are related to dihedrals elsewhere in the protein.

    Compare each pocket residue's dihedrals to all dihedrals in the protein.  Find all pocket
    dihedrals who are related (using mutual information or Pearson coefficient) to other
    dihedrals.  If a residue pair has multiple correlated dihedrals that correlate, only show the
    most-correlated dihedral pair.

    Parameters
    ----------
    pocket_resindex_list : list (or 1D array) of integers
        The 0-indexed residue index assigned by MDAnalysis to each residue of the pocket.
    all_resindex_list : list (or 1D array) of integers
        The 0-indexed residue index assigned by MDAnalysis to each residue of the protein.
    dict_all_res_diheds : dictionary
        The output of `get_dihedrals_for_resindex_list` for all residues in `all_resindex_list`.
    u : MDAnalysis universe
        The universe object that the data are taken from.
    neighbors_to_exclude : integer
        The number of neighbors on each side of the pocket atom to exclude.  E.g. if
        `neighbors_to_exclude=0` then a pocket residue will be compared to its neighbors; if
        `neighbors_to_exclude=1` then the pocket residue won't be compared to the residues on
        either side of it.
    score_cutoff : float
        Only show dihedral pairings whose score (Pearson or mutual information) exceeds
        `score_cutoff`.
    score : string
        Either 'pearson' or 'mut_inf'.  Controls whether Pearson correlation coefficient or mutual
        information is used.

    Returns
    -------
    dict_pocket_resindex_to_scores : dictionary
        This stores Pearson coefficients or mutual informations of pairs of dihedrals that are
        highly related.  Each key is an index (0-indexed; assigned by MDAnalysis) of a pocket
        residue.  Each value is a list of two lists.  The first contains strings labeling
        which dihedrals are being compared; the second contains the score for that pair.
    """

    dict_pocket_resindex_to_scores = {}

    for pocket_resindex in pocket_resindex_list:
        max_dihed_labels = []
        max_mut_infs = []
        print("Checking pocket residue", pocket_resindex)
        for resindex in all_resindex_list:
            residue = u.residues[resindex]
            scores_this_res_pair = []
            dihed_labels_this_res_pair = []
            if abs(residue.resindex - pocket_resindex) <= neighbors_to_exclude:
                continue
            for pocket_dihed_name, pocket_dihed_vals in dict_all_res_diheds[pocket_resindex].items():
                for dihed_name, dihed_vals in dict_all_res_diheds[residue.resindex].items():
                    if (pocket_dihed_vals is None) or (dihed_vals is None):
                        continue
                    if score == "pearson":
                        score_val = scipy.stats.pearsonr(pocket_dihed_vals, dihed_vals)[0]
                    elif score == "mut_inf":
                        pocket_dihed_vals_reshaped = pocket_dihed_vals.reshape(-1, 1)
                        score_val = sklearn.feature_selection.mutual_info_regression(pocket_dihed_vals_reshaped,
                                                                                 dihed_vals)[0]
                    else:
                        raise ValueError("Score must be 'pearson' or 'mut_inf'")
                    label = "%d_%s_vs_%d_%s" %(pocket_resindex, pocket_dihed_name,
                                               residue.resindex, dihed_name)
                    dihed_labels_this_res_pair.append(label)
                    scores_this_res_pair.append(score_val)
            scores_array = np.array(scores_this_res_pair)
            max_score = max(scores_array)
            if max_score > score_cutoff:
                max_score_index = np.argmax(scores_array)
                max_mut_infs.append(max_score)
                max_mut_inf_label = dihed_labels_this_res_pair[max_score_index]
                max_dihed_labels.append(max_mut_inf_label)
        dict_pocket_resindex_to_scores[pocket_resindex] = [max_dihed_labels, np.array(max_mut_infs)]
    return dict_pocket_resindex_to_scores


def graph_high_scores(dict_pocket_resindex_to_score, dihed_dict, rcparams={}):
    """
    Graph each high-scoring dihedral pair.
   
    Create a KDE plot comparing each high-scoring dihedral pair.  Although this function
    provides a quick way to inspect data, publication-quality graphs may require additional
    customization.
   
    Parameters
    ----------
    dict_pocket_resindex_to_score : dictionary
        The output of `get_key_pocket_dihedrals`
    dihed_dict : dictionary
        The output of `get_dihedrals_for_resindex_list`
    rcparams : dictionary
        Parameters passed to matplotlib.pyplot's rcparams.  E.g. rcparams={"figure.figsize":(3,3)}
        would set figures to be 3x3.
    """

    for param, value in rcparams.items():
        plt.rcParams[param] = value

    for pocket_resindex, scores in dict_pocket_resindex_to_score.items():
        for i in range(len(scores[0])):
            short_label = scores[0][i]
            score = scores[1][i]

            pocket_dihedral = short_label.split("_")[1]
            other_resindex = int(short_label.split("_")[2])
            other_dihedral = short_label.split("_")[3]

            pocket_dihed_vals = dihed_dict[pocket_resindex][pocket_dihedral]
            other_dihed_vals = dihed_dict[other_resindex][other_dihedral]

            print("score:", score)
            sns.kdeplot(x=pocket_dihed_vals, y=other_dihed_vals, fill=True,
                        cmap="gnuplot2_r")
            sns.kdeplot(x=pocket_dihed_vals, y=other_dihed_vals, color="black")
            plt.xlabel("0-Indexed Residue %d %s" %(pocket_resindex, pocket_dihedral))
            plt.ylabel("0-Indexed Residue %d %s" %(other_resindex, other_dihedral))
            plt.show()
