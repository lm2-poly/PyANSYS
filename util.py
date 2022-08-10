import numpy as np
import scipy as spy

#------------------ Exceptions ------------------
#----------------------------------------------------------





#------------------ MAPDL - XPL ------------------
#----------------------------------------------------------

def nb_modes(xpl):
    dic = xpl.json()
    list_informations = dic['children']
    dic = list_informations[3]
    list_of_modes = dic['children']
    return len(list_of_modes)


#------------------ Matrixs treatment ------------------
#----------------------------------------------------------

def from_unsym_to_sym(M):
    diag = spy.sparse.diags(M.diagonal())
    return M + M.transpose() - diag

def compute_modal_coeff(matrix, eigen_vector):
    coeff = eigen_vector.T.dot(matrix.dot(eigen_vector)) \
        / eigen_vector.T.dot(eigen_vector)
    return coeff

#------------------Eigen value following ------------------
#----------------------------------------------------------
def diff_cost(freqs_1, freqs_2):
    n_1, n_2 = np.size(freqs_1), np.size(freqs_2)
    cost_matrix = np.zeros((n_1, n_2))
    for i, freq_1 in enumerate(freqs_1):
        for j, freq_2 in enumerate(freqs_2):
            cost_matrix[i,j] = abs(freq_1 - freq_2)
    return cost_matrix

#------------------Added mass computation------------------
#----------------------------------------------------------

def freq_to_ev(freq):
    return (2 * np.pi * freq) ** 2











#---------------------Polar coordinates---------------------
#-----------------------------------------------------------
def cart2pol(x, y, positive_only=True):
    """Converts 2D cartesian coordinates to polar coordinates

    Parameters
    ----------
    x (float): x coordinate of point
    y (float): y coordinate of point
    positive_only (boolean): to have angles from 0 to 2*pi rather than -pi to pi

    Returns
    -------
    radius (float): calculated radius of point
    angle (float): calculated angle of point

    """
    radius = np.linalg.norm([x,y])
    angle = np.arctan2(y, x)
    if positive_only:
        if angle < 0:
            angle = 2*np.pi + angle

    return (radius, angle)

def cart2pol_array(U, positive_only = True):
    """
    Applies cart2pol to an array.
    """
    X = U[:,0]
    Y = U[:,1]
    (n_ligns,) = np.shape(X)
    U_pol = np.zeros((n_ligns, 2))
    for i in range(n_ligns):
        U_pol[i,0], U_pol[i,1] = cart2pol(X[i], Y[i], positive_only)
    return U_pol