"""
# single integrator
"""
from casadi import*
import math
import time
import scipy.io
import numpy as np
import numpy.matlib
from casadi.tools import *
# from shift import shift
from rk4 import rk4
from plotfn import realplot,plot
from spline import path_spline,waypoint_generator
from matplotlib import pyplot as plt
import rospy, tf
import geometry_msgs.msg
from nav_msgs.msg import Odometry

global feedback_states
feedback_states = [0.,0.,0.,0.,0.,0.,0.]

# Initialize our node
rospy.init_node('bebopcontrol',anonymous=True)

# Setup publisher
msg = geometry_msgs.msg.Twist()
pub = rospy.Publisher('bebop/cmd_vel',geometry_msgs.msg.Twist, queue_size=2)

def bebopOdomCallback(message):

    pos = message
    quat = pos.pose.pose.orientation
    # From quaternion to Euler
    angles = tf.transformations.euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
    th = angles[2]
    pose = [pos.pose.pose.position.x, pos.pose.pose.position.y, pos.pose.pose.position.z]  # X, Y, Z 
    vel = [pos.twist.twist.linear.x, pos.twist.twist.linear.y, pos.twist.twist.linear.z]
    
    feedback_states[0] = pose[0]
    feedback_states[1] = pose[1]
    feedback_states[2] = pose[2]
    feedback_states[3] = vel[0]
    feedback_states[4] = vel[1]
    feedback_states[5] = vel[2]
    feedback_states[6] = th

    # Reporting
    # print('bebopOdomCallback: x=%4.1f,y=%4.1f'%(pose[0],pose[1]))

def shift_gazebo(dt, t0, x0, u, f):
    con = np.copy(u[0,:].T)
    
    theta = feedback_states[6]
    vx_I = con[0]
    vy_I = con[1]

    # Publish
    msg.linear.x = vx_I*cos(theta)+vy_I*sin(theta)
    msg.linear.y = -vx_I*sin(theta)+vy_I*cos(theta)
    pub.publish(msg)

    # Setup subscription 
    rospy.Subscriber('bebop/odom',Odometry,bebopOdomCallback)

    x0[0] = feedback_states[0]
    x0[1] = feedback_states[1]

    t0 = t0 + dt
    u0 = np.copy(u[1:,:])
    u0 = np.vstack((u0, np.array(u[-1,:])))
    
    # rospy.sleep(0.1)

    return t0,x0,u0 

ref_traj = np.load('circle.npy')

dt = 0.1        # sampling time [s]
N = 5          # prediction horizon
sim_tim = 100 #500    # Maximum simulation time

states = struct_symSX(["x","y"]) # prediction model
n_states = states.size # Number of states
x,y = states[...]

controls = struct_symSX(["vx","vy"]) # control vector of the system
n_controls = controls.size
vx,vy = controls[...]
rhs = struct_SX(states)
rhs["x"] = vx 
rhs["y"] = vy

f = Function('f',[states,controls],[rhs]) # nonlinear mapping function f(x,u)
U = SX.sym("U",n_controls,N) # Decision variables (controls)
P = SX.sym("P",n_states+(N*2)+n_controls) # parameters which include the initial state, reference points, and last control input
X = SX.sym("X",n_states,(N+1)) # A Matrix that represents the states over the optimization problem.

R = np.identity(2) # control weight
obj = 0 # Objective function
g = []  # constraints vector
st = X[:,0] # initial state
g.append(st - P[0:n_states]) # initial condition constraints
# compute solution symbolically
for k in range(N):
    st = X[:,k]
    con = U[:,k]
    xr = P[n_states+k]
    yr = P[n_states+N+k]
    obj = obj+((st[0]-xr)**2 + (st[1]-yr)**2 + con.T@R@con ) # calculate obj
    st_next = X[:,k+1]
    st_next_rk4 = rk4(f,st,con,dt) #RK4
    g.append(st_next - st_next_rk4) # compute constraints

#add delta_control cost
O_U = SX.sym("O_U",n_controls,1) # previous control
O_U[0,0] = P[n_states+2*N]
O_U[1,0] = P[n_states+2*N+1]
obj = obj + ((U[:,0]-O_U).T@R@(U[:,0]-O_U))

# make the decision variables one column vector
OPT_variables = vertcat(reshape(X,n_states*(N+1),1),reshape(U,n_controls*N,1))
g = vertcat(*g)
#print(g.shape)
jit_options = {"flags": ["-Ofast -march=native"], "verbose": True}
nlp_prob = {'x': OPT_variables,'f': obj,'g': g, 'p': P}
# opts = {"print_time": False,"ipopt.print_level":0,"ipopt.max_iter":10,"ipopt.acceptable_tol":1e-8,"ipopt.acceptable_obj_change_tol":1e-6,"ipopt.warm_start_init_point":"yes"}
opts = {"print_time": False,"ipopt.linear_solver":"ma97","ipopt.print_level":0,"ipopt.max_iter":10,"ipopt.acceptable_tol":1e-8,"ipopt.acceptable_obj_change_tol":1e-6,"ipopt.warm_start_init_point":"yes","jit": True, "compiler": "shell", "jit_options": jit_options}

solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

# Equality Constraints (Multiple Shooting)
lbg = np.zeros(n_states*(N+1))
ubg = np.zeros(n_states*(N+1))
# State Constraints
lbx = np.zeros((n_states*(N+1)+ n_controls*N,1))
ubx = np.zeros((n_states*(N+1)+ n_controls*N,1))
# Create the indices list for constraints, states and controls
xIndex = np.arange(0, n_states*(N+1), n_states).tolist()
yIndex = np.arange(1, n_states*(N+1), n_states).tolist()

vxIndex = np.arange(n_states*(N+1),n_states*(N+1)+ n_controls*N, n_controls).tolist()
vyIndex = np.arange(n_states*(N+1)+1,n_states*(N+1)+ n_controls*N, n_controls).tolist()
# Feed Bounds For State Constraints
lbx[xIndex,:]       = -inf
lbx[yIndex,:]       = -inf

ubx[xIndex,:]       = inf
ubx[yIndex,:]       = inf
# Feed Bounds For control Constraints
v_max = 0.3
v_min = -v_max

lbx[vxIndex,:] = v_min
lbx[vyIndex,:] = v_min

ubx[vxIndex,:] = v_max
ubx[vyIndex,:] = v_max

#print(lbx)
#print(ubx)

#----------------------------------------------
# ALL OF THE ABOVE IS JUST A PROBLEM SETTING UP


# THE SIMULATION LOOP SHOULD START FROM HERE
#-------------------------------------------
t0 = 0.0
x_i = 0.0
y_i = 0.0

x0 = np.array([x_i,y_i])  # initial condition.
xx = np.copy(x0) # xx contains the history of states
t = np.copy(t0)

u0 = np.zeros((N,n_controls))  # two control inputs
X0 = np.matlib.repmat(x0, 1, N+1) # initialization of the states decision variables
X0 = X0.T

# initialize reference
cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec, xd_dot, yd_dot = path_spline(ref_traj[0,:], ref_traj[1,:])
x_waypoints, y_waypoints, phi_Waypoints, xdot_ref, ydot_ref = waypoint_generator(x_i,y_i,ref_traj[0,:],ref_traj[1,:],arc_vec,cs_x_path,cs_y_path,cs_phi_path,arc_length,xd_dot,yd_dot,N)

# Start MPC
mpciter = 0

np.save('/home/msi/Documents/UT/Drone_class/code/data/ref.npy',ref_traj)

# the main simulaton loop...
main_loop = time.time()
rate = rospy.Rate(10)
# while mpciter < sim_tim / dt:
# while True:
while not rospy.is_shutdown():
    # current_time = mpciter*dt
    loop_time = time.time()
    p0 = np.copy(x0)
    p00 = vertcat(p0,x_waypoints,y_waypoints,u0[0,:])
    x00 = vertcat(reshape(X0.T,n_states*(N+1),1), reshape(u0.T,n_controls*N,1))
    solver_time = time.time()
    sol = solver(x0= x00, lbx= lbx, ubx= ubx,lbg= lbg, ubg= ubg,p=p00)
    sol_elapsed = time.time() - solver_time
    solution = sol['x'].full()
    control = np.copy(solution[n_states*(N+1):])
    u = np.copy(reshape(control.T,n_controls,N).T)
    states = np.copy(solution[0:n_states*(N+1)])
    # print(u[0,:])
    # if mpciter == 0:
    #     uu = np.array(u[0,:])
    #     xx1 = np.array(((reshape(states.T,n_states,N+1)).full()).T)
    # else:                                            # history of controls, pred vel, and predicted states
    #     uu = vertcat(uu,u[0,:])
    #     xx1 = vertcat(xx1,((reshape(states.T,n_states,N+1)).full()).T)
    t0,x0,u0 = shift_gazebo(dt,t0,x0,u,f) # get the initialization of the next optimization step
    x_waypoints, y_waypoints, phi_Waypoints, xdot_ref, ydot_ref = waypoint_generator(x0[0],x0[1],ref_traj[0,:],ref_traj[1,:],arc_vec,cs_x_path,cs_y_path,cs_phi_path,arc_length,xd_dot,yd_dot,N)
    # t = vertcat(t,t0)
    # xx = vertcat(xx,x0)
    X0 = np.copy(reshape(states.T,n_states,N+1).T)   # Shift trajectory to initialize the next step
    X0_first = np.copy(X0[1:,:])
    X0 = np.vstack((X0_first, np.array(X0[-1,:])))
    # print(mpciter)
    # print(current_time)
    # print(time.time() - loop_time)
    mpciter = mpciter + 1
    # realplot(xx,n_states,t,t0,xx1,mpciter,N,ref_traj[0,:],ref_traj[1,:])
    # xx = xx.full()
    # t = t.full()
    # np.save('/home/msi/Documents/UT/Drone_class/code/data/x.npy',xx)
    # np.save('/home/msi/Documents/UT/Drone_class/code/data/t.npy',t)
    # rospy.sleep(0.1)
    loop_elapsed = time.time() - loop_time
    rate.sleep()
    print("solver= %s  ,  loop= %s" % (sol_elapsed, loop_elapsed))

elapsed = time.time() - main_loop
print('end of loop')

average_mpc_time = elapsed/(mpciter)
print(average_mpc_time)

t = t.full()
xx = xx.full()
uu = uu.full()
xx1 = xx1.full()
# plot(xx,n_states,t,t0,xx1,mpciter,N,ref_traj[0,:],ref_traj[1,:])
