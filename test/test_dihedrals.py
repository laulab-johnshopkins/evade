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
                         [(0, 0, [162.84396238, -161.34961879, 157.72267142]),
                          (2, 1, [165.71839726, 176.87725266, 163.96622462])])

def test_psi(dihedral_arr_index, res_index, expected_vals):
    all_dihedral_labels, all_dihedrals = scoops.dihedrals.get_dihedrals_for_resindex_list(u.residues.resindices[0:3], u, stop=3)
    psi_sel = u.residues[res_index].psi_selection()
    psi_obj = mda.analysis.dihedrals.Dihedral([psi_sel]).run(stop=3)
    psi_angles= psi_obj.results.angles.T[0]
    np.testing.assert_allclose(psi_angles, all_dihedrals[dihedral_arr_index])
    np.testing.assert_allclose(psi_angles, np.array(expected_vals))
    assert all_dihedral_labels[dihedral_arr_index] == "%d_psi" %(res_index)


# Similar to above, but it uses chi1 and the sort-by-dihedral option.
@pytest.mark.parametrize("dihedral_arr_index, res_index, expected_vals",
                         [(5, 2, [-166.55996348, -142.58900354,  170.78627515])])
def test_chi1_sort_by_dihedrals(dihedral_arr_index, res_index, expected_vals):
    all_dihedral_labels, all_dihedrals = scoops.dihedrals.get_dihedrals_for_resindex_list(u.residues.resindices[0:3], u, stop=3, sort_by="dihedral")
    chi1_sel = u.residues[res_index].chi1_selection()
    chi1_obj = mda.analysis.dihedrals.Dihedral([chi1_sel]).run(stop=3)
    chi1_angles= chi1_obj.results.angles.T[0]
    np.testing.assert_allclose(chi1_angles, all_dihedrals[dihedral_arr_index])
    np.testing.assert_allclose(chi1_angles, np.array(expected_vals))
    assert all_dihedral_labels[dihedral_arr_index] == "%d_chi1" %(res_index)


@pytest.mark.parametrize("dihedral_arr_index, expected_vals",
                         [(0, [ 1., -0.20784814, 0.9999479, -0.52663338, -0.84702695, 0.92410978, 0.98792825]),
                          (2, [ 0.9999479, -0.19785213, 1., -0.53528378, -0.84155669, 0.92796243, 0.98629541])])

def test_circ_corr(dihedral_arr_index, expected_vals):
    all_dihedral_labels, all_dihedrals = scoops.dihedrals.get_dihedrals_for_resindex_list(u.residues.resindices[0:3], u, stop=3)
    score_matrix = scoops.dihedrals.get_dihedral_score_matrix(all_dihedrals, "circ_corr")
    # Construct a list of circular correlation coefficients from Astropy.  This list should be the
    # same as a row of the get_dihedrals_for_resindex_list output.
    this_dihedral_scores = []
    for i in range(len(all_dihedral_labels)):
        a_score = astropy.stats.circcorrcoef(np.radians(all_dihedrals[dihedral_arr_index]), np.radians(all_dihedrals[i]))
        this_dihedral_scores.append(a_score)
    np.testing.assert_allclose(score_matrix[dihedral_arr_index], this_dihedral_scores)
    np.testing.assert_allclose(score_matrix[dihedral_arr_index], expected_vals)


@pytest.mark.parametrize("dihedral_arr_index, expected_vals",
                         [(1, [0., 0.99563492, 0.01765873, 0., 0., 0., 0.]),
                          (3, [0.00456349, 0., 0., 0.99563492, 0.0093254, 0.03789683, 0.00956349])])

def test_mut_inf(dihedral_arr_index, expected_vals):
    all_dihedral_labels, all_dihedrals = scoops.dihedrals.get_dihedrals_for_resindex_list(u.residues.resindices[0:3], u)

    all_dihedrals = (all_dihedrals + 360) % 360

    score_matrix = scoops.dihedrals.get_dihedral_score_matrix(all_dihedrals, "mut_inf")
    mut_inf = sklearn.feature_selection.mutual_info_regression(np.radians(all_dihedrals.T), np.radians(all_dihedrals[dihedral_arr_index]))
    np.testing.assert_allclose(score_matrix[dihedral_arr_index], expected_vals, rtol=1e-5)
    np.testing.assert_allclose(score_matrix[dihedral_arr_index], expected_vals, rtol=1e-5)