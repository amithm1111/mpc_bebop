from scipy.interpolate import CubicSpline
import numpy as np

vd = 3 #0.3
dt = 0.1

def path_spline(x_path, y_path):
    x_diff = np.diff(x_path)
    y_diff = np.diff(y_path)
    phi = np.unwrap(np.arctan2(y_diff, x_diff))
    phi_init = phi[0]
    phi = np.hstack(( phi_init, phi  ))
    arc = np.cumsum( np.sqrt( x_diff**2+y_diff**2 )   )
    arc_length = arc[-1]
    arc_vec = np.linspace(0, arc_length, np.shape(x_path)[0])
    cs_x_path = CubicSpline(arc_vec, x_path)
    cs_y_path = CubicSpline(arc_vec, y_path)
    cs_phi_path = CubicSpline(arc_vec, phi)
    xd_dot = cs_x_path.derivative()
    yd_dot = cs_y_path.derivative()
    return cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec, xd_dot, yd_dot

def waypoint_generator(x_global_init, y_global_init, x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length, xd_dot, yd_dot,N):
    idx = np.argmin( np.sqrt((x_global_init-x_path_data)**2+(y_global_init-y_path_data)**2))
    arc_curr = arc_vec[idx]
    arc_pred = arc_curr + vd*dt
    arc_look = np.linspace(arc_curr, arc_pred, N)
    x_waypoints = cs_x_path(arc_look)
    y_waypoints =  cs_y_path(arc_look)
    phi_Waypoints = cs_phi_path(arc_look)
    xdot_ref = xd_dot(arc_look)
    ydot_ref = yd_dot(arc_look)
    return x_waypoints, y_waypoints, phi_Waypoints, xdot_ref, ydot_ref