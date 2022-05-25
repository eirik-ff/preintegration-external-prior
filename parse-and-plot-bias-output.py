import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

    def ypr_as_deg(self) -> np.ndarray:
        return self.ypr * 180 / np.pi


def rmse(gt_time, gt_data, est_time, est_data):
    """
    gt_time: Nx1 array
    gt_data: Nx3 array 
    est_time: Nx1
    est_data: Nx3
    """
    gt = pd.DataFrame()
    gt["time"] = gt_time
    gt["gt_x"] = gt_data[:, 0]
    gt["gt_y"] = gt_data[:, 1]
    gt["gt_z"] = gt_data[:, 2]

    est = pd.DataFrame()
    est["time"] = est_time
    est["est_x"] = est_data[:, 0]
    est["est_y"] = est_data[:, 1]
    est["est_z"] = est_data[:, 2]

    merged = pd.merge_asof(est, gt, on="time", direction="nearest")
    
    err_x = merged["est_x"] - merged["gt_x"]
    err_y = merged["est_y"] - merged["gt_y"]
    err_z = merged["est_z"] - merged["gt_z"]

    rmse_x = np.sqrt(np.sum(err_x.to_numpy() ** 2) / err_x.size)
    rmse_y = np.sqrt(np.sum(err_y.to_numpy() ** 2) / err_y.size)
    rmse_z = np.sqrt(np.sum(err_z.to_numpy() ** 2) / err_z.size)

    return np.array([rmse_x, rmse_y, rmse_z])


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["vicon", "leica_full", "leica_skip40"], help="Which dataset to use")
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

if args.dataset.startswith("leica"):
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

    print("gt", gt.time.shape)


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
print("data", data.time.shape)

if args.dataset.startswith("leica"):
    # this is to make the plot nicer since we're using ypr
    R_leica = Rot.from_matrix([
        [0, 0, -1],
        [0, -1, 0],
        [1, 0, 0]
    ])
    data.ypr = (Rot.from_euler("ZYX", data.ypr) * R_leica).as_euler("ZYX")
    gt.ypr = (Rot.from_euler("ZYX", gt.ypr) * R_leica).as_euler("ZYX")
elif args.dataset == "vicon":
    # fix alignment that isn't good for some reason
    data.ypr[:, 2] -= data.ypr[0, 2] - gt.ypr[0, 2]


# rmse
pos_rmse = rmse(gt.time, gt.pos, data.time, data.pos)
ypr_rmse = rmse(gt.time, gt.ypr, data.time, data.ypr) * 180/np.pi
b_a_rmse = rmse(gt.time, gt.bias_acc, data.time, data.bias_acc)
b_g_rmse = rmse(gt.time, gt.bias_gyr, data.time, data.bias_gyr)
if args.dataset == "leica_skip40":
    b_a_rmse_cut25 = rmse(gt.time[gt.time > 25], gt.bias_acc[gt.time > 25], data.time[data.time > 25], data.bias_acc[data.time > 25])
    b_g_rmse_cut25 = rmse(gt.time[gt.time > 25], gt.bias_gyr[gt.time > 25], data.time[data.time > 25], data.bias_gyr[data.time > 25])

pos_rmse_str = '\t'.join(f"{d:.4f}" for d in pos_rmse)
ypr_rmse_str = '\t'.join(f"{d:.4f}" for d in ypr_rmse)
b_a_rmse_str = '\t'.join(f"{d:.4f}" for d in b_a_rmse)
b_g_rmse_str = '\t'.join(f"{d:.4f}" for d in b_g_rmse)
print(f"RMSE: '{args.dataset}'\n pos [m]     = {pos_rmse_str}\n ypr [deg]   = {ypr_rmse_str}\n b_a [m/s^2] = {b_a_rmse_str}\n b_g [rad/s] = {b_g_rmse_str}")

if args.dataset == "leica_skip40":
    b_a_rmse_cut25_str = '\t'.join(f"{d:.4f}" for d in b_a_rmse_cut25)
    b_g_rmse_cut25_str = '\t'.join(f"{d:.4f}" for d in b_g_rmse_cut25)
    print(f" b_a_cut25 [m/s^2] = {b_a_rmse_cut25_str}\n b_g_cut25 [rad/s] = {b_g_rmse_cut25_str}")


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
plt.tight_layout()
plt.savefig("figures/" + args.dataset + "_pos.png", dpi=320)

plt.figure()
ax = plt.subplot(3, 1, 1)
plt.title("Roll, pitch, yaw")
plt.plot(gt.time, gt.ypr_as_deg()[:, 2], "g")
plt.plot(data.time, data.ypr_as_deg()[:, 2], "b")
ax.set_xticklabels([])
plt.ylabel("roll [deg]")
ax = plt.subplot(3, 1, 2)
plt.plot(gt.time, gt.ypr_as_deg()[:, 1], "g")
plt.plot(data.time, data.ypr_as_deg()[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("pitch [deg]")
ax = plt.subplot(3, 1, 3)
plt.plot(gt.time, gt.ypr_as_deg()[:, 0], "g")
plt.plot(data.time, data.ypr_as_deg()[:, 0], "b")
plt.ylabel("yaw [deg]")
plt.xlabel("time [sec]")
plt.tight_layout()
plt.savefig("figures/" + args.dataset + "_ori.png", dpi=320)

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
    plt.tight_layout()
    plt.savefig("figures/" + args.dataset + "_quat.png", dpi=320)

plt.figure()
ax = plt.subplot(3, 1, 1)
plt.title("Bias accelerometer")
plt.plot(gt.time, gt.bias_acc[:, 0], "g")
plt.plot(data.time, data.bias_acc[:, 0], "b")
plt.plot(data.time, np.mean(data.bias_acc[:, 0]) * np.ones_like(data.time), "k--")
ax.set_xticklabels([])
plt.ylabel("x [m/s^2]")
ax = plt.subplot(3, 1, 2)
plt.plot(gt.time, gt.bias_acc[:, 1], "g")
plt.plot(data.time, data.bias_acc[:, 1], "b")
plt.plot(data.time, np.mean(data.bias_acc[:, 1]) * np.ones_like(data.time), "k--")
ax.set_xticklabels([])
plt.ylabel("y [m/s^2]")
ax = plt.subplot(3, 1, 3)
plt.plot(gt.time, gt.bias_acc[:, 2], "g")
plt.plot(data.time, data.bias_acc[:, 2], "b")
plt.plot(data.time, np.mean(data.bias_acc[:, 2]) * np.ones_like(data.time), "k--")
plt.ylabel("z [m/s^2]")
plt.xlabel("time [sec]")
plt.tight_layout()
plt.savefig("figures/" + args.dataset + "_bias_accel.png", dpi=320)

plt.figure()
ax = plt.subplot(3, 1, 1)
plt.title("Bias gyroscope")
plt.plot(gt.time, gt.bias_gyr[:, 0], "g")
plt.plot(data.time, data.bias_gyr[:, 0], "b")
plt.plot(data.time, np.mean(data.bias_gyr[:, 0]) * np.ones_like(data.time), "k--")
ax.set_xticklabels([])
plt.ylabel("x [rad/s]")
ax = plt.subplot(3, 1, 2)
plt.plot(gt.time, gt.bias_gyr[:, 1], "g")
plt.plot(data.time, data.bias_gyr[:, 1], "b")
plt.plot(data.time, np.mean(data.bias_gyr[:, 1]) * np.ones_like(data.time), "k--")
ax.set_xticklabels([])
plt.ylabel("y [rad/s]")
ax = plt.subplot(3, 1, 3)
plt.plot(gt.time, gt.bias_gyr[:, 2], "g")
plt.plot(data.time, data.bias_gyr[:, 2], "b")
plt.plot(data.time, np.mean(data.bias_gyr[:, 2]) * np.ones_like(data.time), "k--")
plt.ylabel("z [rad/s]")
plt.xlabel("time [sec]")
plt.tight_layout()
plt.savefig("figures/" + args.dataset + "_bias_gyro.png", dpi=320)

if args.dataset.startswith("leica"):
    plt.figure()
    ax = plt.subplot(3, 1, 1)
    plt.title("Bias accelerometer zoomed in")
    plt.plot(gt.time, gt.bias_acc[:, 0], "g")
    plt.plot(data.time, data.bias_acc[:, 0], "b")
    plt.plot(data.time, np.mean(data.bias_acc[:, 0]) * np.ones_like(data.time), "k--")
    if args.dataset == "leica_skip40":
        plt.ylim([-0.4, 0.3])  # for leica skip40
        plt.plot(data.time[data.time > 25], np.mean(data.bias_acc[data.time > 25, 0]) * np.ones_like(data.time[data.time > 25]), "m--")
    else:
        plt.ylim([-0.25, 0.2])  # for leica full
    ax.set_xticklabels([])
    plt.ylabel("x [m/s^2]")
    ax = plt.subplot(3, 1, 2)
    plt.plot(gt.time, gt.bias_acc[:, 1], "g")
    plt.plot(data.time, data.bias_acc[:, 1], "b")
    if args.dataset == "leica_skip40":
        plt.ylim([-0.3, 0.5])  # for leica skip40
        plt.plot(data.time[data.time > 25], np.mean(data.bias_acc[data.time > 25, 1]) * np.ones_like(data.time[data.time > 25]), "m--")
    else:
        plt.ylim([-0.2, 0.5])  # for leica full
    ax.set_xticklabels([])
    plt.ylabel("y [m/s^2]")
    ax = plt.subplot(3, 1, 3)
    plt.plot(gt.time, gt.bias_acc[:, 2], "g")
    plt.plot(data.time, data.bias_acc[:, 2], "b")
    if args.dataset == "leica_skip40":
        plt.ylim([-0.7, 0.3])  # for leica skip40
        plt.plot(data.time[data.time > 25], np.mean(data.bias_acc[data.time > 25, 2]) * np.ones_like(data.time[data.time > 25]), "m--")
    else:
        plt.ylim([-0.4, 0.5])  # for leica full
    plt.ylabel("z [m/s^2]")
    plt.xlabel("time [sec]")
    plt.tight_layout()
    plt.savefig("figures/" + args.dataset + "_bias_accel_zoom.png", dpi=320)

#plt.tight_layout()
plt.show()

