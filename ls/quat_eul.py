from transforms3d.euler import quat2euler
from transforms3d.euler import euler2quat
from geometry_msgs.msg import PoseStamped

def eul2quat(theta):
    orientation = PoseStamped().pose.orientation
    orientation.w, orientation.x, orientation.y, orientation.z  = euler2quat(0, 0, theta)
    return orientation

def quat2eul(orientation):
    _,_,theta = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z])
    return theta #Convertion to radians