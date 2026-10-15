"""Camera frame (forward, left, up) -> world frame (east, north, up) using GPS + inertial unit."""
import numpy as np


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def camera_rotation(roll, pitch, yaw, mount_pitch):
    """World rotation of the camera: body orientation, then the fixed camera tilt."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll) @ rot_y(mount_pitch)


def cam_to_world(p_cam, cam_pos, rotation):
    return np.asarray(cam_pos) + rotation @ np.asarray(p_cam)
