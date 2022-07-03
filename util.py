#------------------Added mass computation------------------
#----------------------------------------------------------




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