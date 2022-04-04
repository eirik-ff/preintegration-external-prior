#include <iostream>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <gtsam_unstable/slam/PartialPriorFactor.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

#define USE_ISAM 0

std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> preint;
gtsam::NonlinearFactorGraph graph;
gtsam::Values initial;
gtsam::ISAM2 isam;

auto ext_pos_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-3);
auto est_vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-1);
auto ext_pose_noise = gtsam::noiseModel::Isotropic::Sigmas(
    (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3).finished() // rad, rad, rad, m, m, m
);

int imu_meas_since_last_optim_count = 0;
double prev_imu_timestamp = 0;
gtsam::NavState prev_state;
gtsam::imuBias::ConstantBias prev_bias;

double prev_ext_pos_timestamp = 0;
gtsam::Point3 prev_ext_pos;

int ext_count = 0;

void imgCallback(const sensor_msgs::Image::ConstPtr msg)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exceptin : %s", e.what());
    return;
  }

  cv::imshow("cam0", cv_ptr->image);
  cv::waitKey(1);
}

void imuCallback(const sensor_msgs::Imu::ConstPtr msg)
{
  static int imu_meas_count = 0;
  std::cout << "Got IMU measurement " << ++imu_meas_count << std::endl;

  double timestamp = msg->header.stamp.toSec();
  auto &a = msg->linear_acceleration;
  auto &g = msg->angular_velocity;

  gtsam::Vector3 acc(a.x, a.y, a.z);
  gtsam::Vector3 gyr(g.x, g.y, g.z);

  if (prev_imu_timestamp == 0)
  {
    prev_imu_timestamp = timestamp;
    std::cout << "Set first ever prev imu timestamp" << std::endl;
  }
  else
  {
    imu_meas_since_last_optim_count++;
    double dt = timestamp - prev_imu_timestamp;
    prev_imu_timestamp = timestamp;
    preint->integrateMeasurement(acc, gyr, dt);

    std::cout << "Preintegrated measurement, deltas: \n"
              << "  deltaTij     = " << preint->deltaTij() << "\n"
              << "  deltaPij     = " << preint->deltaPij().transpose() << "\n"
              << "  deltaVij     = " << preint->deltaVij().transpose() << "\n"
              << "  deltaRij.rpy = " << preint->deltaRij().rpy().transpose() << "\n"
              << "--------------------" << std::endl;
  }
}

void extPosCallback(double timestamp, const gtsam::Point3 *ext_pos, const gtsam::Rot3 *ext_rot)
{
  if (imu_meas_since_last_optim_count == 0)
  {
    return;
  }

  // std::cout << "Got external position callback." << std::endl;
  ext_count++;

  if (ext_count == 1)
  {
    prev_ext_pos_timestamp = timestamp;
    prev_ext_pos = *ext_pos;

    if (ext_rot)
    {
      gtsam::Pose3 prior(*ext_rot, *ext_pos);
      graph.addPrior(X(ext_count), prior, ext_pose_noise);

      prev_state = gtsam::NavState(prior, gtsam::Vector3::Zero());
      prev_bias = gtsam::imuBias::ConstantBias(gtsam::Vector6::Zero());
    }
    else
    {
      gtsam::PartialPriorFactor<gtsam::Pose3> prior_factor(X(ext_count), {3, 4, 5}, *ext_pos, ext_pos_noise);
      graph.add(prior_factor);

      prev_state = gtsam::NavState(gtsam::Pose3(gtsam::Rot3::identity(), *ext_pos), gtsam::Vector3::Zero());
      prev_bias = gtsam::imuBias::ConstantBias(gtsam::Vector6::Zero());
    }

    initial.insert(X(ext_count), prev_state.pose());
    initial.insert(V(ext_count), prev_state.velocity());
    initial.insert(B(ext_count), prev_bias);

    std::cout << "First ever ext pos, add prior and initial values" << std::endl;
    return;
  }

  gtsam::CombinedImuFactor imu_factor(X(ext_count - 1), V(ext_count - 1),
                                      X(ext_count), V(ext_count),
                                      B(ext_count - 1), B(ext_count),
                                      *preint);
  graph.add(imu_factor);

  if (ext_rot)
  {
    gtsam::Pose3 prior(*ext_rot, *ext_pos);
    graph.addPrior(X(ext_count), prior, ext_pose_noise);
  }
  else
  {
    gtsam::PartialPriorFactor<gtsam::Pose3> prior_factor(X(ext_count), {3, 4, 5}, *ext_pos, ext_pos_noise);
    graph.add(prior_factor);
  }

  // double dt = timestamp - prev_ext_pos_timestamp;
  // prev_ext_pos_timestamp = timestamp;
  // gtsam::Vector3 est_vel_prior = (pos - prev_ext_pos) / dt;
  // gtsam::PriorFactor<gtsam::Vector3> vel_prior_factor(V(ext_count), est_vel_prior, est_vel_noise);
  // graph.add(vel_prior_factor);

  gtsam::NavState prop_state = preint->predict(prev_state, prev_bias);
  initial.insert(X(ext_count), prop_state.pose());
  initial.insert(V(ext_count), prop_state.velocity());
  initial.insert(B(ext_count), prev_bias);

  // PRINT
  // graph.print("FACTOR GRAPH: ");
  // initial.print("INITIAL VALUES: ");

  // OPTIMIZE
#if USE_ISAM
  isam.update(graph, initial);
  isam.update();

  graph.resize(0);
  initial.clear();

  gtsam::Pose3 x = isam.calculateEstimate<gtsam::Pose3>(X(ext_count));
  gtsam::Vector3 v = isam.calculateEstimate<gtsam::Vector3>(V(ext_count));
  gtsam::imuBias::ConstantBias b = isam.calculateEstimate<gtsam::imuBias::ConstantBias>(B(ext_count));

  prev_state = gtsam::NavState(x, v);
  prev_bias = b;
#else
  gtsam::LevenbergMarquardtParams param;
  param.setVerbosityLM("SUMMARY");
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, param);
  gtsam::Values result = optimizer.optimize();
  // gtsam::Marginals marginals(graph, result);
  std::cout << std::endl;
  initial.update(result);

  // SAVE RESULTS
  prev_state = gtsam::NavState(result.at<gtsam::Pose3>(X(ext_count)), result.at<gtsam::Vector3>(V(ext_count)));
  prev_bias = result.at<gtsam::imuBias::ConstantBias>(B(ext_count));
#endif

  imu_meas_since_last_optim_count = 0;

  preint->resetIntegrationAndSetBias(prev_bias);

  // PRINT LATEST RESULT
  std::cout << "--------------------\nRESULTS: " << std::endl;
  std::cout << "Prior from mocap: \n"
            << "  pos = " << ext_pos->transpose() << std::endl;
  if (ext_rot)
  {
    std::cout << "  rot = " << ext_rot->rpy().transpose() << std::endl;
  }
  std::cout << "State: \n"
            << "  pos = " << prev_state.pose().translation().transpose() << "\n"
            << "  rot = " << prev_state.attitude().rpy().transpose() << "\n"
            << "  vel = " << prev_state.velocity().transpose() << std::endl;
  // std::cout << "Uncertainty for position: \n"
  //           << marginals.marginalCovariance(X(ext_count)) << std::endl;
  // std::cout << "Uncertainty for velocity: \n"
  //           << marginals.marginalCovariance(V(ext_count)) << std::endl;
  std::cout << "Error against external prior: \n"
            << "  pos = " << (*ext_pos - prev_state.pose().translation()).transpose() << "\n";
  if (ext_rot)
  {
    std::cout << "  rot = " << ext_rot->between(prev_state.attitude()).rpy().transpose() << std::endl;
  }
  std::cout << "Bias: " << prev_bias << std::endl;
  // std::cout << "Uncertainty for bias: \n"
  //           << marginals.marginalCovariance(B(ext_count)) << std::endl;
  std::cout << "--------------------" << std::endl;
}

void leicaCallback(const geometry_msgs::PointStamped::ConstPtr msg)
{
  double timestamp = msg->header.stamp.toSec();
  gtsam::Point3 ext_pos(msg->point.x, msg->point.y, msg->point.z);

  extPosCallback(timestamp, &ext_pos, nullptr);
}

void viconCallback(const geometry_msgs::TransformStamped::ConstPtr msg)
{
  static int count = 0;
  count++;
  std::cout << "Got Vicon measurement " << count << std::endl;
  // only process every 5 message since vicon runs at 100 hz instead of 20 hz (like leica)
  if (count % 5)
  {
    double timestamp = msg->header.stamp.toSec();
    gtsam::Quaternion quat(msg->transform.rotation.w, msg->transform.rotation.x, msg->transform.rotation.y, msg->transform.rotation.z);
    gtsam::Rot3 ext_rot(quat);
    gtsam::Point3 ext_pos(msg->transform.translation.x, msg->transform.translation.y, msg->transform.translation.z);

    extPosCallback(timestamp, &ext_pos, &ext_rot);
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_node");
  ros::NodeHandle nh;

  // auto p = boost::make_shared<gtsam::PreintegrationCombinedParams>(gtsam::Vector3(0, 0, -9.81));
  auto p = gtsam::PreintegrationCombinedParams::MakeSharedU();
  // p->accelerometerCovariance = gtsam::I_3x3 * 0.008 * 0.008;
  // p->gyroscopeCovariance = gtsam::I_3x3 * 0.0012 * 0.0012;
  // p->biasAccCovariance = gtsam::I_3x3 * 0.1 * 0.1;
  // p->biasOmegaCovariance = gtsam::I_3x3 * 0.03 * 0.03;
  p->accelerometerCovariance = gtsam::I_3x3 * 2e-3 * 2e-3;
  p->gyroscopeCovariance = gtsam::I_3x3 * 1.6968e-04 * 1.6968e-04;
  p->biasAccCovariance = gtsam::I_3x3 * 3e-3 * 3e-3;
  p->biasOmegaCovariance = gtsam::I_3x3 * 1.9393e-05 * 1.9393e-05;

  p->integrationCovariance = gtsam::I_3x3 * 1e-4 * 1e-4;
  p->biasAccOmegaInt = gtsam::I_6x6 * 1e-4 * 1e-4;

  // FOR EUROC MACHINE ROOM (T_imu_leica = T_sensor_body)
  // // gtsam::Rot3 rot(0, 0, 1,
  // //                 0, -1, 0,
  // //                 1, 0, 0);
  // // gtsam::Rot3 rot = gtsam::Rot3::identity();
  // gtsam::Point3 trans(7.48903e-02, -1.84772e-02, -1.20209e-01);
  // ros::Subscriber extPosSub = nh.subscribe<geometry_msgs::PointStamped>("/leica/position", 100, leicaCallback);

  // FOR EUROC VICON ROOM (T_imu_vicon = T_sensor_body)
  gtsam::Rot3 rot(0.33638, -0.01749,  0.94156,
                 -0.02078, -0.99972, -0.01114,
                  0.94150, -0.01582, -0.33665);
  gtsam::Point3 trans(0.06901, -0.02781, -0.12395);
  ros::Subscriber extPosSub = nh.subscribe<geometry_msgs::TransformStamped>("/vicon/firefly_sbx/firefly_sbx", 100, viconCallback);

  p->body_P_sensor = gtsam::Pose3(rot, trans).inverse();
  preint = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(p);

  ros::Subscriber imuSub = nh.subscribe<sensor_msgs::Imu>("/imu0", 100, imuCallback);
  // ros::Subscriber imgSub = nh.subscribe<sensor_msgs::Image>("/cam0/image_raw", 10, imgCallback);

  std::cout << "INITIALIZED AND READY." << std::endl;
  ros::spin();
}
