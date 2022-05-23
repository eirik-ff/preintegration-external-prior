import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

from scipy.spatial.transform import Rotation as Rot
from dataclasses import dataclass
from typing import Optional

@dataclass
class Data:
    time: np.ndarray  # Nx1
    pos: np.ndarray  # Nx3, [x, y, z]
    bias_acc: np.ndarray  # Nx3, [x, y, z]
    bias_gyr: np.ndarray  # Nx3, [x, y, z]
    ypr: np.ndarray = None  # Nx3, [yaw, pitch, roll]
    quat: np.ndarray = None  # Nx4, [x, y, z, w]

    def transf_pos_with_calib(self, t_imu_mocap: np.ndarray) -> np.ndarray:
        return self.pos - t_imu_mocap

    def ypr_from_quat(self, R_imu_mocap: Rot) -> np.ndarray:
        if self.quat is not None:
            ypr =  Rot.from_quat(self.quat) * R_imu_mocap
            return ypr.as_euler("ZYX")




parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["vicon", "leica"], help="Which dataset to use")
parser.add_argument("data", help="Path to data file")
parser.add_argument("--gt", help="Path to ground truth file")
parser.add_argument("--quat", action="store_true", help="Include this if dataset include quaternion output")

args = parser.parse_args()


pattern_ypr_only = r"RESULTS: t = (-?[\d.]+(?:e-?\d+)?)\s.+\s.+\s.*\s?State: \s  pos = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  ypr = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  vel = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s.+\s.+\s.*\s?Bias: \s  acc = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  gyr = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)"
pattern_quat = r"RESULTS: t = (-?[\d.]+(?:e-?\d+)?)\s.+\s.+\s.*\s?State: \s  pos = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  ypr = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  vel = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  quat = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s.+\s.+\s.*\s?Bias: \s  acc = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  gyr = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)"

if args.quat:
    pattern = pattern_quat
else:
    pattern = pattern_ypr_only

if args.dataset == "leica":
    # R_IL = Rot.from_matrix(np.array([
    #     0, 0, -1,
    #     0, -1, 0,
    #     1, 0, 0
    # ]).reshape((3,3)))
    R_IL = Rot.from_matrix(np.eye(3))
    t_IL = np.array([7.48903e-02, -1.84772e-02, -1.20209e-01])

    R_imu_mocap = R_IL
    t_imu_mocap = t_IL

elif args.dataset == "vicon":
    R_IV = Rot.from_matrix(np.array([
        0.33638, -0.01749,  0.94156,
        -0.02078, -0.99972, -0.01114,
        0.94150, -0.01582, -0.33665
    ]).reshape((3,3)))
    t_IV = np.array([0.06901, -0.02781, -0.12395])

    R_imu_mocap = R_IV
    t_imu_mocap = t_IV


gt: Data = None
if args.gt:
    gt_filename = sys.argv[2]
    gt_data = np.genfromtxt(args.gt, delimiter=",", skip_header=1)
    
    gt_time = gt_data[:, 0]
    gt_pos = gt_data[:, 1:4]
    gt_quat = np.array([gt_data[:, 5], gt_data[:, 6], gt_data[:, 7], gt_data[:, 4]]).T
    gt_vel = gt_data[:, 8:11]
    gt_bias_acc = gt_data[:, 14:17]
    gt_bias_gyr = gt_data[:, 11:14]

    gt = Data(gt_time, gt_pos, gt_bias_acc, gt_bias_gyr, quat=gt_quat)
    gt.pos = gt.transf_pos_with_calib(t_imu_mocap)
    gt.ypr = gt.ypr_from_quat(R_imu_mocap)
    gt.time /= 1e9


filename = args.data
with open(filename, "r") as f:
    string = f.read()

if "quat" in string and not args.quat:
    print("You provided a dataset with quaternion data but didn't use --quat flag. Use --quat flag.")
    sys.exit(1)

matches = re.findall(pattern, string)

data = []
#for (t, px, py, pz, yaw, pitch, roll, vx, vy, vz, bax, bay, baz, bgx, bgy, bgz) in matches:
for d in matches:
    data.append([ float(i) for i in d ])


data = np.array(data)

# convert to seconds since start
gt.time -= data[0, 0]
data[:, 0] -= data[0, 0]

# alises
if args.quat:
    time = data[:, 0]
    pos = data[:, 1:4]
    ypr = data[:, 4:7]
    vel = data[:, 7:10]
    quat = np.array([data[:, 11], data[:, 12], data[:, 13], data[:, 10]]).T
    b_a = data[:, 14:17]
    b_g = data[:, 17:20]
else: 
    time = data[:, 0]
    pos = data[:, 1:4]
    ypr = data[:, 4:7]
    vel = data[:, 7:10]
    b_a = data[:, 10:13]
    b_g = data[:, 13:16]
    quat = None

data = Data(time, pos, b_a, b_g, ypr=ypr, quat=quat)

if args.dataset == "leica":
    # this is to make the plot nicer since we're using ypr
    R_leica = Rot.from_matrix([
        [0, 0, -1],
        [0, -1, 0],
        [1, 0, 0]
    ])
    data.ypr = (Rot.from_euler("ZYX", data.ypr) * R_leica).as_euler("ZYX")
    gt.ypr = (Rot.from_euler("ZYX", gt.ypr) * R_leica).as_euler("ZYX")

# plot 
sns.set()

plt.figure()
ax = plt.subplot(3, 1, 1)
plt.title("Position")
plt.plot(gt.time, gt.pos[:, 0], "g")
plt.plot(data.time, data.pos[:, 0], "b")
ax.set_xticklabels([])
plt.ylabel("x [m]")
ax = plt.subplot(3, 1, 2)
plt.plot(gt.time, gt.pos[:, 1], "g")
plt.plot(data.time, data.pos[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("y [m]")
ax = plt.subplot(3, 1, 3)
plt.plot(gt.time, gt.pos[:, 2], "g")
plt.plot(data.time, data.pos[:, 2], "b")
plt.ylabel("z [m]")
plt.xlabel("time [sec]")

plt.figure()
ax = plt.subplot(3, 1, 1)
plt.title("Roll, pitch, yaw")
plt.plot(gt.time, gt.ypr[:, 2], "g")
plt.plot(data.time, data.ypr[:, 2], "b")
ax.set_xticklabels([])
plt.ylabel("roll [rad]")
ax = plt.subplot(3, 1, 2)
plt.plot(gt.time, gt.ypr[:, 1], "g")
plt.plot(data.time, data.ypr[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("pitch [rad]")
ax = plt.subplot(3, 1, 3)
plt.plot(gt.time, gt.ypr[:, 0], "g")
plt.plot(data.time, data.ypr[:, 0], "b")
plt.ylabel("yaw [rad]")
plt.xlabel("time [sec]")

if args.quat:
    plt.figure()
    ax = plt.subplot(4, 1, 1)
    plt.title("Quaternion")
    plt.plot(gt.time, gt.quat[:, 0], "g")
    plt.plot(data.time, data.quat[:, 0], "b")
    ax.set_xticklabels([])
    plt.ylabel("x")
    ax = plt.subplot(4, 1, 2)
    plt.plot(gt.time, gt.quat[:, 1], "g")
    plt.plot(data.time, data.quat[:, 1], "b")
    ax.set_xticklabels([])
    plt.ylabel("y")
    ax = plt.subplot(4, 1, 3)
    plt.plot(gt.time, gt.quat[:, 2], "g")
    plt.plot(data.time, data.quat[:, 2], "b")
    ax.set_xticklabels([])
    plt.ylabel("z")
    ax = plt.subplot(4, 1, 4)
    plt.plot(gt.time, gt.quat[:, 3], "g")
    plt.plot(data.time, data.quat[:, 3], "b")
    plt.ylabel("w")
    plt.xlabel("time [sec]")

plt.figure()
ax = plt.subplot(3, 1, 1)
plt.title("Bias accelerometer")
plt.plot(gt.time, gt.bias_acc[:, 0], "g")
plt.plot(data.time, data.bias_acc[:, 0], "b")
ax.set_xticklabels([])
plt.ylabel("x [m/s^2]")
ax = plt.subplot(3, 1, 2)
plt.plot(gt.time, gt.bias_acc[:, 1], "g")
plt.plot(data.time, data.bias_acc[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("y [m/s^2]")
ax = plt.subplot(3, 1, 3)
plt.plot(gt.time, gt.bias_acc[:, 2], "g")
plt.plot(data.time, data.bias_acc[:, 2], "b")
plt.ylabel("z [m/s^2]")
plt.xlabel("time [sec]")

plt.figure()
ax = plt.subplot(3, 1, 1)
plt.title("Bias gyroscope")
plt.plot(gt.time, gt.bias_gyr[:, 0], "g")
plt.plot(data.time, data.bias_gyr[:, 0], "b")
ax.set_xticklabels([])
plt.ylabel("x [rad/s]")
ax = plt.subplot(3, 1, 2)
plt.plot(gt.time, gt.bias_gyr[:, 1], "g")
plt.plot(data.time, data.bias_gyr[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("y [rad/s]")
ax = plt.subplot(3, 1, 3)
plt.plot(gt.time, gt.bias_gyr[:, 2], "g")
plt.plot(data.time, data.bias_gyr[:, 2], "b")
plt.ylabel("z [rad/s]")
plt.xlabel("time [sec]")

#plt.tight_layout()
plt.show()

