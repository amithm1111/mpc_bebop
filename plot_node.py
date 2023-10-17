import numpy as np
from plotfn import realplot,plot
from matplotlib import pyplot as plt
import rospy, tf
import geometry_msgs.msg
from nav_msgs.msg import Odometry

global states
states = [0.,0.,0.,0.,0.,0.,0.]
st_x = []
st_y = []

ref = np.load('circle.npy')

# Initialize our node
rospy.init_node('plot_node',anonymous=True)

def Callback(message):

    pos = message
    pose = [pos.pose.pose.position.x, pos.pose.pose.position.y, pos.pose.pose.position.z]  # X, Y, Z 
    # vel = [pos.twist.twist.linear.x, pos.twist.twist.linear.y, pos.twist.twist.linear.z]
    
    states[0] = pose[0]
    states[1] = pose[1]
    # feedback_states[2] = pose[2]
    # feedback_states[3] = vel[0]
    # feedback_states[4] = vel[1]
    # feedback_states[5] = vel[2]
    # feedback_states[6] = th
    # Reporting
    # print('bebopOdomCallback: x=%4.1f,y=%4.1f'%(pose[0],pose[1]))

while True:
    # Setup subscription 
    rospy.Subscriber('bebop/odom',Odometry,Callback)

    st_x.append(states[0])
    st_y.append(states[1])
    # sx = np.array(st_x)
    # sy = np.array(st_y)
    # np.save('/home/msi/Documents/UT/Drone_class/code/data/sx.npy',sx)
    # np.save('/home/msi/Documents/UT/Drone_class/code/data/sy.npy',sy)
     
    fig = plt.figure(2)
    plt.plot(ref[0,:],ref[1,:],'--g',linewidth=2)
    plt.plot(st_x,st_y,'b',linewidth=2)
    plt.pause(1e-10)
    plt.draw()

    rospy.sleep(0.01)
