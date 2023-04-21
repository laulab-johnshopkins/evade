import math
import numpy as np
import scipy.stats
import MDAnalysis as mda
import MDAnalysis.analysis.dihedrals
from multiprocess import Pool
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.feature_selection
import tqdm


def get_dihedrals_for_resindex_list(resindex_list, u, step=None, start=None, stop=None,
                                    dihedrals_to_include=["phi", "psi", "chi1", "chi2"],
                                    sort_by = "resindex"):
    """
    Get the dihedrals for all residue across an MD trajectory.

    This function gets the phi, psi, chi1, and chi2 dihedrals for all residues.
    Note that some angles aren't defined for some residues.  E.g. the phi and psi angles
    may not be present at the ends of chains, and certain residues don't have chi1 and chi2.

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
    dihedrals_to_include : list, optional
        Controls which dihedrals are calculated.  The default is `["phi", "psi", "chi1", "chi2"]`.
    sort_by : string, optional
        Controls how the dihedrals are ordered in the output.  The default is `"resindex"`, which
        groups the dihedrals by residue.  (E.g. the code would return
        `["5_phi", "5_psi", "6_phi", "6_psi"]` if only phi and
        psi are calculated.)  The alternative is `sort_by = "dihedral"`.  This lists all the phis,
        then all the psis, etc.

    Returns
    -------
    all_dihedral_labels : list
        The ordered list of which dihedrals are stored in `all_dihedrals`.  Entries are formatted
        like `"5_chi1"` where the number is the 0-indexed residue number assigned by MDAnalysis.
        Note that the numbers in `all_dihedral_labels` are likely to differ from the residue
        indices used by the MD data's topology file.
    all_dihedrals : numpy array
        An n-by-m array, where n is the number of dihedrals and m is the number of trajectory frames.
        Angles are in radians and range from -pi to +pi.
    """
    
    # If `dihedrals_to_include` omits any of these, then they intentionally stay empty.
    residues_with_phi = []
    residues_with_psi = []
    residues_with_chi1 = []
    residues_with_chi2 = []
    
    if "phi" in dihedrals_to_include:
        # Get list of residues that have phi dihedral.
        phi_sel_with_none = u.residues[resindex_list].phi_selections()
        for i in range(len(phi_sel_with_none)):
            if phi_sel_with_none[i] is not None:
                residues_with_phi.append(resindex_list[i])
        # Get an array of phi dihedrals.  Each row is the result for a residue.
        phi_sel_without_none = list(filter(None, phi_sel_with_none))
        phi_object = mda.analysis.dihedrals.Dihedral(phi_sel_without_none).run(step=step, start=start,
                                                                               stop=stop)
        phi_angles = phi_object.results.angles.T
    
    if "psi" in dihedrals_to_include:
        # Get list of residues that have psi dihedral.
        psi_sel_with_none = u.residues[resindex_list].psi_selections()
        for i in range(len(psi_sel_with_none)):
            if psi_sel_with_none[i] is not None:
                residues_with_psi.append(resindex_list[i])
        # Get an array of psi dihedrals.  Each row is the result for a residue.
        psi_sel_without_none = list(filter(None, psi_sel_with_none))
        psi_object = mda.analysis.dihedrals.Dihedral(psi_sel_without_none).run(step=step, start=start,
                                                                               stop=stop)
        psi_angles = psi_object.results.angles.T
    
    if "chi1" in dihedrals_to_include:
        # Get list of residues that have chi1 dihedral.
        chi1_sel_with_none = u.residues[resindex_list].chi1_selections()
        for i in range(len(chi1_sel_with_none)):
            if chi1_sel_with_none[i] is not None:
                residues_with_chi1.append(resindex_list[i])
        # Get an array of chi1 dihedrals.  Each row is the result for a residue.
        chi1_sel_without_none = list(filter(None, chi1_sel_with_none))
        chi1_object = mda.analysis.dihedrals.Dihedral(chi1_sel_without_none).run(step=step, start=start,
                                                                                 stop=stop)
        chi1_angles = chi1_object.results.angles.T  
    
    if "chi2" in dihedrals_to_include:
        # Get chi2 dihedrals.  MDAnalysis only finds chi2 through the Janin function.
        chi2_angles = []
        # The select_remove argument is an MDAnalysis default value.  I included it in case MDAnalysis's
        # default ever changes, to reduce the likelihood of the code breaking.
        removed_residues = "resname ALA CYS* GLY PRO SER THR VAL"
        janin_object = mda.analysis.dihedrals.Janin(u.residues[resindex_list].atoms,
                                                    select_remove=removed_residues).run(step=step,
                                                                                        start=start,
                                                                                        stop=stop)
        janin_angles = janin_object.results.angles
        # The Janin function returns angles between 0 and 360.  The standard chi1 and chi2
        # definitions allow values between -180 and 180.
        vectorized_math_remainder = np.vectorize(math.remainder)
        janin_angles = vectorized_math_remainder(janin_angles, 360)
        # Determine which residues have chi2.
        janin_index = 0 # track how many residues taken from Janin data
        for residue in u.residues[resindex_list]:
            # This removes all the residues that the `select_remove` argument removed from the Janin
            # dataset.  These are the residues for which chi2 isn't defined.
            if ((residue.resname not in ["ALA", "GLY", "CYS", "SER", "THR", "VAL", "PRO"]) and 
                (residue.resname[0:3] != "CYS")):
                chi2_angles_this_res = janin_angles[:,janin_index][:,1]
                residues_with_chi2.append(residue.resindex)
                chi2_angles.append(chi2_angles_this_res)
                janin_index += 1

    if sort_by == "resindex":
        # Combine all dihedrals into a single list.  The list is ordered by residue.
        all_dihedrals = []
        all_dihedral_labels = []
        for resindex in resindex_list:
            if resindex in residues_with_phi:
                phi_index = residues_with_phi.index(resindex)
                phi_vals = phi_angles[phi_index]
                all_dihedrals.append(phi_vals)
                all_dihedral_labels.append("%d_phi" %(resindex))
            if resindex in residues_with_psi:
                psi_index = residues_with_psi.index(resindex)
                psi_vals = psi_angles[psi_index]
                all_dihedrals.append(psi_vals)
                all_dihedral_labels.append("%d_psi" %(resindex))
            if resindex in residues_with_chi1:
                chi1_index = residues_with_chi1.index(resindex)
                chi1_vals = chi1_angles[chi1_index]
                all_dihedrals.append(chi1_vals)
                all_dihedral_labels.append("%d_chi1" %(resindex))
            if resindex in residues_with_chi2:
                chi2_index = residues_with_chi2.index(resindex)
                chi2_vals = chi2_angles[chi2_index]
                all_dihedrals.append(chi2_vals)
                all_dihedral_labels.append("%d_chi2" %(resindex))
    
    elif sort_by == "dihedral":
        all_dihedrals = []
        all_dihedral_labels = []
        for dihedral in dihedrals_to_include:
            if dihedral == "phi":
                for resindex in resindex_list:
                    if resindex in residues_with_phi:
                        phi_index = residues_with_phi.index(resindex)
                        phi_vals = phi_angles[phi_index]
                        all_dihedrals.append(phi_vals)
                        all_dihedral_labels.append("%d_phi" %(resindex))
            elif dihedral == "psi":
                for resindex in resindex_list:
                    if resindex in residues_with_psi:
                        psi_index = residues_with_psi.index(resindex)
                        psi_vals = psi_angles[psi_index]
                        all_dihedrals.append(psi_vals)
                        all_dihedral_labels.append("%d_psi" %(resindex))
            elif dihedral == "chi1":
                for resindex in resindex_list:
                    if resindex in residues_with_chi1:
                        chi1_index = residues_with_chi1.index(resindex)
                        chi1_vals = chi1_angles[chi1_index]
                        all_dihedrals.append(chi1_vals)
                        all_dihedral_labels.append("%d_chi1" %(resindex))
            elif dihedral == "chi2":
                for resindex in resindex_list:
                    if resindex in residues_with_chi2:
                        chi2_index = residues_with_chi2.index(resindex)
                        chi2_vals = chi2_angles[chi2_index]
                        all_dihedrals.append(chi2_vals)
                        all_dihedral_labels.append("%d_chi2" %(resindex))
    else:
        raise ValueError("sort_by must be resindex or dihedral")
                
    all_dihedrals_array = np.array(all_dihedrals)
    all_dihedrals_radians = np.radians(all_dihedrals_array)
    return all_dihedral_labels, all_dihedrals_radians


def get_dihedral_score_matrix(dihed_vals, score):
    """
    Get a matrix containing relatedness scores between dihedrals.

    This function does an all-against-all comparison using a specified scoring metric.

    Parameters
    ----------
    dihed_vals : numpy array
        The output of `get_dihedrals_for_resindex_list`.
    score : string
        Either 'inv_cov' or 'covariance' or 'circ_corr' or 'mut_inf.  Controls which
        score metric is used.

    Returns
    -------
    score_matrix : numpy array
        A square matrix containing scores for pairs of dihedrals.
    """

    if score == "inv_cov" or score == "covariance" or score == "circ_corr":
        # Linear covariance for a sample of size N is the sum of (x - x_mean)(y - y_mean) / (N - 1).
        # For circular covariance, the sine is taken so the covariance is 
        # the sum of sin(x - x_mean) * sin(y - y_mean) / (N - 1).
        dihed_averages = scipy.stats.circmean(dihed_vals, axis=1, low=0, high=2*np.pi)

        # This subtraction is simpler than the one used in the Liu/Amaral/Keten code.  Once
        # the sine is taken, the two results are identical.
        angles_minus_averages = dihed_vals - dihed_averages[:,None]
        
        sin_angles_minus_avg = np.sin(angles_minus_averages)
        
        num_frames = len(dihed_vals[0])
        covariance_matrix = np.matmul(sin_angles_minus_avg, sin_angles_minus_avg.T) / (num_frames-1)
    if score == "inv_cov":
        inverse_covariance_matrix = np.linalg.pinv(covariance_matrix)
        score_matrix = inverse_covariance_matrix
        return score_matrix
    elif score == "covariance":
        score_matrix = covariance_matrix
        return score_matrix
    elif score == "circ_corr":
        '''
        # The correlation coefficient is the covariance divided by the product of each variable's
        # standard deviation.  See https://en.wikipedia.org/wiki/Covariance_and_correlation.
        stds = astropy.stats.circstats.circstd(dihed_vals, axis=1) ** 2
        std_products = np.outer(stds, stds)
        corr_matrix = covariance_matrix / std_products
        return corr_matrix'''
        variances = np.diagonal(covariance_matrix)
        #stds = np.sqrt(variances)
        #std_products = np.outer(stds, stds)
        variance_products = np.outer(variances, variances)
        std_products = np.sqrt(variance_products)
        
        corr_matrix = covariance_matrix / std_products
        return corr_matrix

    elif score == "mut_inf":
        dihed_vals_transpose = np.array(dihed_vals).T
        
        def update_bar(task):
            # See https://stackoverflow.com/questions/71968890/multiprocessing-with-map-async-and-progress-bar
            pbar.update()

        def mut_inf_for_row(row_index):
            return sklearn.feature_selection.mutual_info_regression(dihed_vals_transpose, dihed_vals[row_index])

        num_angles = len(dihed_vals)
        # Pool() defaults to use number of processes equal to os.cpu_count().
        with Pool() as pool:
            with tqdm.tqdm(total=num_angles) as pbar:
                # The callback function is called each time the pool returns a result.
                async_results = [pool.apply_async(mut_inf_for_row, args=(x,), callback=update_bar)
                                 for x in range(num_angles)]
                mut_inf_matrix = [async_result.get() for async_result in async_results]

        return mut_inf_matrix
