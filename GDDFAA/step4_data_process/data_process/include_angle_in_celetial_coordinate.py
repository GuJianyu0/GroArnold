import numpy as np

convert_radian_to_degree = 180./np.pi
convert_degree_to_radian = 1./convert_radian_to_degree

def included_angle_in_spherical_coordinate(phi1, theta1, phi2, theta2, is_unit_degree=False):
    phi1_, theta1_, phi2_, theta2_ = phi1, theta1, phi2, theta2
    if is_unit_degree:
        phi1_, theta1_, phi2_, theta2_ = \
            phi1*convert_degree_to_radian, theta1*convert_degree_to_radian, \
            phi2*convert_degree_to_radian, theta2*convert_degree_to_radian
    print(phi1_, theta1_, phi2_, theta2_)
    cos_alpha = np.cos(phi1_-phi2_)*np.sin(theta1_)*np.sin(theta2_) + np.cos(theta1_)*np.cos(theta2_)
    alpha = np.arccos(cos_alpha)
    if not is_unit_degree:
        return alpha
    else:
        return alpha*convert_radian_to_degree

def included_angle_in_celestial_coordinate(ra1, dec1, ra2, dec2, is_unit_degree=False):
    phi1, theta1, phi2, theta2 = ra1, np.pi/2-dec1, ra2, np.pi/2-dec2
    if is_unit_degree:
        phi1, theta1, phi2, theta2 = ra1, 90.-dec1, ra2, 90.-dec2
    return included_angle_in_spherical_coordinate(phi1, theta1, phi2, theta2, is_unit_degree)

if __name__=="__main__":

    # A1 = [50., -30.]
    # A2 = [60., -70.]
    A1 = [50., -30.]
    A2 = [60., -70.]
    alpha_p = included_angle_in_celestial_coordinate(
        A1[0]*convert_degree_to_radian, A1[1]*convert_degree_to_radian, 
        A2[0]*convert_degree_to_radian, A2[1]*convert_degree_to_radian, 
        is_unit_degree=False
    )
    alpha_d = included_angle_in_celestial_coordinate(
        A1[0], A1[1], A2[0], A2[1], 
        is_unit_degree=True
    )
    print(alpha_p, alpha_d, alpha_d*convert_degree_to_radian)
