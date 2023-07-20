import pytest
import os
import MDAnalysis as mda
import numpy as np
import sklearn
import scoops
import astropy.stats

# Initialize a Universe with the test dataset.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
example_data_loc = os.path.join(THIS_DIR, os.pardir, 'example/mpro_short_traj.pdb')
u = mda.Universe(example_data_loc)


# Check that get_dihedrals_for_resindex_list returns correct values
# for psi dihedrals.  The code is checked against MDAnalysis's
# result when run on just that residue, as well as with the result
# given when the tests were written.
@pytest.mark.parametrize("dihedral_arr_index, res_index, expected_vals",
                         [(0, 0, [ 2.84216331, -2.81608209,  2.75277992]),
                          (3, 1, [ 2.89233166,  3.08709043,  2.86175048])])

def test_psi(dihedral_arr_index, res_index, expected_vals):
    dihedral_df = scoops.dihedrals.get_dihedrals_for_resindex_list(u.residues.resindices[0:3], u, stop=3)
    all_dihedrals = dihedral_df.to_numpy()
    all_dihedral_labels = dihedral_df.index.tolist()
    psi_sel = u.residues[res_index].psi_selection()
    psi_obj = mda.analysis.dihedrals.Dihedral([psi_sel]).run(stop=3)
    psi_angles= psi_obj.results.angles.T[0]
    psi_angles = np.radians(psi_angles)
    np.testing.assert_allclose(psi_angles, all_dihedrals[dihedral_arr_index])
    np.testing.assert_allclose(psi_angles, np.array(expected_vals))
    assert all_dihedral_labels[dihedral_arr_index] == "%d_psi" %(res_index)


# Similar to above, but it uses chi1 and the sort-by-dihedral option.
@pytest.mark.parametrize("dihedral_arr_index, res_index, expected_vals",
                         [(6, 2, [-2.90701976, -2.48864759,  2.98078282])])
def test_chi1_sort_by_dihedrals(dihedral_arr_index, res_index, expected_vals):
    dihedral_df = scoops.dihedrals.get_dihedrals_for_resindex_list(u.residues.resindices[0:3], u, stop=3, sort_by="dihedral")
    all_dihedrals = dihedral_df.to_numpy()
    all_dihedral_labels = dihedral_df.index.tolist()
    chi1_sel = u.residues[res_index].chi1_selection()
    chi1_obj = mda.analysis.dihedrals.Dihedral([chi1_sel]).run(stop=3)
    chi1_angles= chi1_obj.results.angles.T[0]
    chi1_angles = np.radians(chi1_angles)
    np.testing.assert_allclose(chi1_angles, all_dihedrals[dihedral_arr_index])
    np.testing.assert_allclose(chi1_angles, np.array(expected_vals))
    assert all_dihedral_labels[dihedral_arr_index] == "%d_chi1" %(res_index)


@pytest.mark.parametrize("dihedral_arr_index, expected_vals",
                         [(0, [ 1.        , -0.4278929 , -0.20784814,  0.9999479 , -0.52663338, -0.84702695,  0.92410978,  0.98792825]),
                          (3, [ 0.9999479 , -0.41864421, -0.19785213,  1.        , -0.53528378, -0.84155669,  0.92796243,  0.98629541])])

def test_circ_corr(dihedral_arr_index, expected_vals):
    dihedral_df = scoops.dihedrals.get_dihedrals_for_resindex_list(u.residues.resindices[0:3], u, stop=3)
    all_dihedrals = dihedral_df.to_numpy()
    all_dihedral_labels = dihedral_df.index.tolist()
    score_df = scoops.dihedrals.get_dihedral_score_matrix(dihedral_df, "circ_corr")
    score_matrix = score_df.to_numpy()
    # Construct a list of circular correlation coefficients from Astropy.  This list should be the
    # same as a row of the get_dihedrals_for_resindex_list output.
    this_dihedral_scores = []
    for i in range(len(all_dihedral_labels)):
        a_score = astropy.stats.circcorrcoef(all_dihedrals[dihedral_arr_index], all_dihedrals[i])
        this_dihedral_scores.append(a_score)
    np.testing.assert_allclose(score_matrix[dihedral_arr_index], this_dihedral_scores)
    np.testing.assert_allclose(score_matrix[dihedral_arr_index], expected_vals)


@pytest.mark.parametrize("dihedral_arr_index, expected_vals",
                         [(2, [0.        , 0.        , 0.99563492, 0.03789683, 0.        ,  0.        , 0.        , 0.        ]),
                          (4, [0.        , 0.        , 0.        , 0.02678571, 0.99563492,  0.0093254 , 0.02083333, 0.        ])])

def test_mut_inf(dihedral_arr_index, expected_vals):
    dihedral_df = scoops.dihedrals.get_dihedrals_for_resindex_list(u.residues.resindices[0:3], u)
    all_dihedrals = dihedral_df.to_numpy()
    all_dihedral_labels = dihedral_df.index.tolist()

    score_df = scoops.dihedrals.get_dihedral_score_matrix(dihedral_df, "mut_inf")
    score_matrix = score_df.to_numpy()
    mut_inf = sklearn.feature_selection.mutual_info_regression(all_dihedrals.T, all_dihedrals[dihedral_arr_index])
    np.testing.assert_allclose(score_matrix[dihedral_arr_index], expected_vals, rtol=1e-5)
    np.testing.assert_allclose(score_matrix[dihedral_arr_index], mut_inf, rtol=1e-5)