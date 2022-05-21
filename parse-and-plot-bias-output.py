import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.transform import Rotation as Rot

#DATASET_NAME = "leica"
DATASET_NAME = "vicon"

pattern = r"RESULTS: t = (-?[\d.]+(?:e-?\d+)?)\s.+\s.+\s.*\s?State: \s  pos = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  ypr = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  vel = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s.+\s.+\s.*\s?Bias: \s  acc = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)\s  gyr = +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?) +(-?[\d.]+(?:e-?\d+)?)"

if len(sys.argv) not in [2, 3]:
    print(f"Usage: {sys.argv[0]} [path to data file] [OPTIONAL: path to ground truth]")
    sys.exit(1)

R_IV = Rot.from_matrix(np.array([
    0.33638, -0.01749,  0.94156,
   -0.02078, -0.99972, -0.01114,
    0.94150, -0.01582, -0.33665
]).reshape((3,3)))
t_IV = np.array([0.06901, -0.02781, -0.12395])

# R_IL = Rot.from_matrix(np.array([
#     0, 0, -1,
#     0, -1, 0,
#     1, 0, 0
# ]).reshape((3,3)))
R_IL = Rot.from_matrix(np.eye(3))
t_IL = np.array([7.48903e-02, -1.84772e-02, -1.20209e-01])

if DATASET_NAME == "leica":
    R_imu_mocap = R_IL
    t_imu_mocap = t_IL
elif DATASET_NAME == "vicon":
    R_imu_mocap = R_IV
    t_imu_mocap = t_IV
else:
    print(f"Unknown dataset name: '{DATASET_NAME}'")
    sys.exit(1)


gt_data = np.zeros((0, 16))
if len(sys.argv) == 3:
    gt_filename = sys.argv[2]
    gt_data_quat = np.genfromtxt(gt_filename, delimiter=",", skip_header=1)

    gt_data = np.zeros((gt_data_quat.shape[0], 16))
    gt_data[:, 0] = gt_data_quat[:, 0]
    gt_data[:, 1:4] = gt_data_quat[:, 1:4] - t_imu_mocap
    # orientation quat of imu, need to transform to body (i.e. mocap)
    quat = np.array([gt_data_quat[:, 5], gt_data_quat[:, 6], gt_data_quat[:, 7], gt_data_quat[:, 4]]).T
    q = Rot.from_quat(quat)
    q = q * R_imu_mocap.inv()
    gt_data[:, 4:7] = q.as_euler("ZYX")
    gt_data[:, 7:10] = gt_data_quat[:, 8:11]
    # bias is already given in imu frame so no need to transform to body (mocap)
    gt_data[:, 13:16] = gt_data_quat[:, 11:14]
    gt_data[:, 10:13] = gt_data_quat[:, 14:17]


filename = sys.argv[1]
with open(filename, "r") as f:
    string = f.read()

matches = re.findall(pattern, string)

data = []
#for (t, px, py, pz, yaw, pitch, roll, vx, vy, vz, bax, bay, baz, bgx, bgy, bgz) in matches:
for d in matches:
    data.append([ float(i) for i in d ])


data = np.array(data)

# convert to seconds since start
gt_data[:, 0] /= 1e9
gt_data[:, 0] -= data[0, 0]
data[:, 0] -= data[0, 0]

# alises
time = data[:, 0]
pos = data[:, 1:4]
ypr = data[:, 4:7]
vel = data[:, 7:10]
b_a = data[:, 10:13]
b_g = data[:, 13:16]

gt_time = gt_data[:, 0]
gt_pos = gt_data[:, 1:4]
gt_ypr = gt_data[:, 4:7]
gt_vel = gt_data[:, 7:10]
gt_b_a = gt_data[:, 10:13]
gt_b_g = gt_data[:, 13:16]

# plot 
sns.set()

plt.figure()
plt.suptitle(DATASET_NAME.capitalize())

ax = plt.subplot(6, 2, 1)
plt.title("Position")
plt.plot(gt_time, gt_pos[:, 0], "g")
plt.plot(time, pos[:, 0], "b")
ax.set_xticklabels([])
plt.ylabel("x [m]")
ax = plt.subplot(6, 2, 3)
plt.plot(gt_time, gt_pos[:, 1], "g")
plt.plot(time, pos[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("y [m]")
ax = plt.subplot(6, 2, 5)
plt.plot(gt_time, gt_pos[:, 2], "g")
plt.plot(time, pos[:, 2], "b")
ax.set_xticklabels([])
plt.ylabel("z [m]")

ax = plt.subplot(6, 2, 7)
plt.title("Roll, pitch, yaw")
plt.plot(gt_time, gt_ypr[:, 2], "g")
plt.plot(time, ypr[:, 2], "b")
ax.set_xticklabels([])
plt.ylabel("roll [rad]")
ax = plt.subplot(6, 2, 9)
plt.plot(gt_time, gt_ypr[:, 1], "g")
plt.plot(time, ypr[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("pitch [rad]")
ax = plt.subplot(6, 2, 11)
plt.plot(gt_time, gt_ypr[:, 0], "g")
plt.plot(time, ypr[:, 0], "b")
plt.ylabel("yaw [rad]")
plt.xlabel("time [sec]")

ax = plt.subplot(6, 2, 2)
plt.title("Bias accelerometer")
plt.plot(gt_time, gt_b_a[:, 0], "g")
plt.plot(time, b_a[:, 0], "b")
ax.set_xticklabels([])
plt.ylabel("x [m/s^2]")
ax = plt.subplot(6, 2, 4)
plt.plot(gt_time, gt_b_a[:, 1], "g")
plt.plot(time, b_a[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("y [m/s^2]")
ax = plt.subplot(6, 2, 6)
plt.plot(gt_time, gt_b_a[:, 2], "g")
plt.plot(time, b_a[:, 2], "b")
ax.set_xticklabels([])
plt.ylabel("z [m/s^2]")

ax = plt.subplot(6, 2, 8)
plt.title("Bias gyroscope")
plt.plot(gt_time, gt_b_g[:, 0], "g")
plt.plot(time, b_g[:, 0], "b")
ax.set_xticklabels([])
plt.ylabel("x [rad/s]")
ax = plt.subplot(6, 2, 10)
plt.plot(gt_time, gt_b_g[:, 1], "g")
plt.plot(time, b_g[:, 1], "b")
ax.set_xticklabels([])
plt.ylabel("y [rad/s]")
ax = plt.subplot(6, 2, 12)
plt.plot(gt_time, gt_b_g[:, 2], "g")
plt.plot(time, b_g[:, 2], "b")
plt.ylabel("z [rad/s]")
plt.xlabel("time [sec]")

#plt.tight_layout()
plt.show()

