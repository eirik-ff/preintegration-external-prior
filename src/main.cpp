#include <iostream>
#include <iomanip>

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
#include <gtsam/navigation/GPSFactor.h>

#include <gtsam_unstable/slam/PartialPriorFactor.h>
#include <gtsam_unstable/nonlinear/FixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

constexpr struct {
    bool imu = false;
    bool vicon = false;
    bool leica = false;
    bool graph = false;
    bool optim = true;
    bool result = true;
} verbosity;

enum class BackendType
{
  LM,   // Levenberg-Marquardt
  ISAM, // Incremental smoothing and mapping
  IFL   // Incremental fixed lag smoother
};

constexpr BackendType backend_type = BackendType::ISAM;

std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> preint;
gtsam::NonlinearFactorGraph graph;
gtsam::Values initial;
gtsam::ISAM2 isam;
gtsam::IncrementalFixedLagSmoother ifl;

gtsam::FixedLagSmootherKeyTimestampMap key_timestamp_map;

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
  if (verbosity.imu)
    std::cout << "Got IMU measurement " << ++imu_meas_count << std::endl;

  double timestamp = msg->header.stamp.toSec();
  auto &a = msg->linear_acceleration;
  auto &g = msg->angular_velocity;

  gtsam::Vector3 acc(a.x, a.y, a.z);
  gtsam::Vector3 gyr(g.x, g.y, g.z);

  if (prev_imu_timestamp == 0)
  {
    prev_imu_timestamp = timestamp;

    if (verbosity.imu)
      std::cout << "Set first ever prev imu timestamp" << std::endl;
  }
  else
  {
    imu_meas_since_last_optim_count++;
    double dt = timestamp - prev_imu_timestamp;
    prev_imu_timestamp = timestamp;
    preint->integrateMeasurement(acc, gyr, dt);

    if (verbosity.imu)
      std::cout << "Preintegrated measurement, deltas: \n"
                << "  deltaTij     = " << preint->deltaTij() << "\n"
                << "  deltaPij     = " << preint->deltaPij().transpose() << "\n"
                << "  deltaVij     = " << preint->deltaVij().transpose() << "\n"
                << "  deltaRij.rpy = " << preint->deltaRij().rpy().transpose() << "\n"
                << "--------------------" << std::endl;
  }
}

void processExtPose(double timestamp, const gtsam::Point3 *ext_pos, const gtsam::Rot3 *ext_rot)
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

    auto velocity_noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1); // m/s
    auto bias_noise_model = gtsam::noiseModel::Isotropic::Sigma(6, 1e-1);

    // zero priors for velocity and bias
    gtsam::Vector3 prior_vel;
    gtsam::imuBias::ConstantBias prior_bias;

    if (ext_rot)
    {
      gtsam::Pose3 prior(*ext_rot, *ext_pos);

      // IMPORTANT to add initial priors on velocity and bias to make the problem
      // well constrained and not run into indeterminant linear system exceptions.
      graph.addPrior(X(ext_count), prior, ext_pose_noise);
      graph.addPrior(V(ext_count), prior_vel, velocity_noise_model);
      graph.addPrior(B(ext_count), prior_bias, bias_noise_model);

      prev_state = gtsam::NavState(prior, prior_vel);
      prev_bias = gtsam::imuBias::ConstantBias(prior_bias);
    }
    else
    {
      // Need full pose prior on first variable
      gtsam::Pose3 prior(gtsam::Rot3::identity(), *ext_pos);
      graph.addPrior(X(ext_count), prior, ext_pose_noise);

      graph.addPrior(V(ext_count), prior_vel, velocity_noise_model);
      graph.addPrior(B(ext_count), prior_bias, bias_noise_model);

      prev_state = gtsam::NavState(gtsam::Pose3(gtsam::Rot3::identity(), *ext_pos), prior_vel);
      prev_bias = prior_bias;
    }

    initial.insert(X(ext_count), prev_state.pose());
    initial.insert(V(ext_count), prev_state.velocity());
    initial.insert(B(ext_count), prev_bias);

    key_timestamp_map.insert({X(ext_count), timestamp});
    key_timestamp_map.insert({V(ext_count), timestamp});
    key_timestamp_map.insert({B(ext_count), timestamp});

    if (backend_type == BackendType::ISAM)
    {
      isam.update(graph, initial);

      graph.resize(0);
      initial.clear();

      gtsam::Pose3 x = isam.calculateEstimate<gtsam::Pose3>(X(ext_count));
      gtsam::Vector3 v = isam.calculateEstimate<gtsam::Vector3>(V(ext_count));
      gtsam::imuBias::ConstantBias b = isam.calculateEstimate<gtsam::imuBias::ConstantBias>(B(ext_count));

      prev_state = gtsam::NavState(x, v);
      prev_bias = b;
    }

    // need to reset preintegration because the very first state x1 is set to be
    // at where-ever it is when the priors are added
    preint->resetIntegrationAndSetBias(prev_bias);

    if (verbosity.vicon || verbosity.leica)
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
    // gtsam::PartialPriorFactor<gtsam::Pose3> prior_factor(X(ext_count), {3, 4, 5}, *ext_pos, ext_pos_noise);
    gtsam::GPSFactor prior_factor(X(ext_count), *ext_pos, ext_pos_noise);
    graph.add(prior_factor);
  }

  key_timestamp_map.insert({X(ext_count), timestamp});
  key_timestamp_map.insert({V(ext_count), timestamp});
  key_timestamp_map.insert({B(ext_count), timestamp});

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
  if (verbosity.graph)
  {
    if (backend_type == BackendType::LM)
    {
      graph.print("FACTOR GRAPH: ");
      initial.print("INITIAL VALUES: ");
    }
    else if (backend_type == BackendType::ISAM)
    {
      graph.print("GRAGH TO ADD TO ISAM: ");
      isam.getFactorsUnsafe().print("CURRENT ISAM FACTORS: ");
    }
  }

  // OPTIMIZE
  std::shared_ptr<gtsam::Marginals> marginals = nullptr;
  if (backend_type == BackendType::LM)
  {
    gtsam::LevenbergMarquardtParams param;
    if (verbosity.optim)
      param.setVerbosityLM("SUMMARY");
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, param);
    gtsam::Values result = optimizer.optimize();
    // gtsam::Marginals marginals(graph, result);
    marginals = std::make_shared<gtsam::Marginals>(graph, result);
    std::cout << std::endl;
    initial.update(result);

    // SAVE RESULTS
    prev_state = gtsam::NavState(result.at<gtsam::Pose3>(X(ext_count)), result.at<gtsam::Vector3>(V(ext_count)));
    prev_bias = result.at<gtsam::imuBias::ConstantBias>(B(ext_count));
  }
  else if (backend_type == BackendType::ISAM)
  {
    isam.update(graph, initial);
    isam.update(); // Not strictly needed

    graph.resize(0);
    initial.clear();

    gtsam::Pose3 x = isam.calculateEstimate<gtsam::Pose3>(X(ext_count));
    gtsam::Vector3 v = isam.calculateEstimate<gtsam::Vector3>(V(ext_count));
    gtsam::imuBias::ConstantBias b = isam.calculateEstimate<gtsam::imuBias::ConstantBias>(B(ext_count));

    prev_state = gtsam::NavState(x, v);
    prev_bias = b;
  }
  else if (backend_type == BackendType::IFL)
  {
    ifl.update(graph, initial, key_timestamp_map);
    ifl.update();

    graph.resize(0);
    initial.clear();

    gtsam::Pose3 x = ifl.calculateEstimate<gtsam::Pose3>(X(ext_count));
    gtsam::Vector3 v = ifl.calculateEstimate<gtsam::Vector3>(V(ext_count));
    gtsam::imuBias::ConstantBias b = ifl.calculateEstimate<gtsam::imuBias::ConstantBias>(B(ext_count));

    prev_state = gtsam::NavState(x, v);
    prev_bias = b;
  }

  imu_meas_since_last_optim_count = 0;

  preint->resetIntegrationAndSetBias(prev_bias);

  // PRINT LATEST RESULT
  if (verbosity.result)
  {
    auto p = std::cout.precision();
    std::cout.precision(17);
    std::cout << "--------------------\nRESULTS: t = " << timestamp << std::endl;
    std::cout.precision(p);
    std::cout << "Prior from mocap: \n"
              << "  pos = " << ext_pos->transpose() << std::endl;
    if (ext_rot)
    {
      std::cout << "  ypr = " << ext_rot->ypr().transpose() << std::endl;
    }
    std::cout << "State: \n"
              << "  pos = " << prev_state.pose().translation().transpose() << "\n"
              << "  ypr = " << prev_state.attitude().ypr().transpose() << "\n"
              << "  vel = " << prev_state.velocity().transpose() << "\n"
              << "  quat = " << prev_state.attitude().quaternion().transpose() << std::endl;
    if (marginals)
    {
      std::cout << "Std for pose: " << marginals->marginalCovariance(X(ext_count)).diagonal().cwiseSqrt().transpose() << std::endl;
      std::cout << "Std for vel: " << marginals->marginalCovariance(V(ext_count)).diagonal().cwiseSqrt().transpose() << std::endl;
    }
    std::cout << "Error against external prior: \n"
              << "  pos = " << (*ext_pos - prev_state.pose().translation()).transpose() << "\n";
    if (ext_rot)
    {
      std::cout << "  ypr = " << ext_rot->between(prev_state.attitude()).ypr().transpose() << std::endl;
    }
    std::cout << "Bias: \n"
              << "  acc = " << prev_bias.accelerometer().transpose() << "\n"
              << "  gyr = " << prev_bias.gyroscope().transpose() << std::endl;
    if (marginals)
      std::cout << "Std for bias: " << marginals->marginalCovariance(B(ext_count)).diagonal().cwiseSqrt().transpose() << std::endl;
    std::cout << "--------------------" << std::endl;
  }
}

void leicaCallback(const geometry_msgs::PointStamped::ConstPtr msg)
{
  static int count = 0;
  count++;

  double timestamp = msg->header.stamp.toSec();
  gtsam::Point3 ext_pos(msg->point.x, msg->point.y, msg->point.z);

  if (verbosity.leica)
    std::cout << "Got Leica measurement " << count << std::endl;

  processExtPose(timestamp, &ext_pos, nullptr);
}

void viconCallback(const geometry_msgs::TransformStamped::ConstPtr msg)
{
  static int count = 0;
  count++;
  if (verbosity.vicon)
    std::cout << "Got Vicon measurement " << count << std::endl;

  // only process every 5 message since vicon runs at 100 hz instead of 20 hz (like leica)
  if (count % 5 == 0)
  {
    double timestamp = msg->header.stamp.toSec();
    gtsam::Quaternion quat(msg->transform.rotation.w, msg->transform.rotation.x, msg->transform.rotation.y, msg->transform.rotation.z);
    gtsam::Rot3 ext_rot(quat);
    gtsam::Point3 ext_pos(msg->transform.translation.x, msg->transform.translation.y, msg->transform.translation.z);

    processExtPose(timestamp, &ext_pos, &ext_rot);
  }
}

int main(int argc, char **argv)
{
  if (argc < 4) // when using roslaunch, ros adds two extra params
  {
    ROS_WARN_STREAM("Missing dataset name\n"
                    << "Usage: " << argv[0] << " [vicon/leica]");
    return 1;
  }
  std::cout << "Using dataset '" << argv[1] << "'" << std::endl;

  ros::init(argc, argv, "test_node");
  ros::NodeHandle nh;

  // auto p = boost::make_shared<gtsam::PreintegrationCombinedParams>(gtsam::Vector3(0, 0, -9.81));
  // auto p = gtsam::PreintegrationCombinedParams::MakeSharedU(9.8082);
  auto p = gtsam::PreintegrationCombinedParams::MakeSharedU(9.77965);
  // FROM SVO EUROC PARAMETER FILE
  // p->accelerometerCovariance = gtsam::I_3x3 * 0.008 * 0.008;
  // p->gyroscopeCovariance = gtsam::I_3x3 * 0.0012 * 0.0012;
  // p->biasAccCovariance = gtsam::I_3x3 * 0.1 * 0.1;
  // p->biasOmegaCovariance = gtsam::I_3x3 * 0.03 * 0.03;
  // FROM EUROC PARAMETER FILE
  p->accelerometerCovariance = gtsam::I_3x3 * 2e-3 * 2e-3;
  p->gyroscopeCovariance = gtsam::I_3x3 * 1.6968e-04 * 1.6968e-04;
  p->biasAccCovariance = gtsam::I_3x3 * 3e-3 * 3e-3;
  p->biasOmegaCovariance = gtsam::I_3x3 * 1.9393e-05 * 1.9393e-05;

  p->integrationCovariance = gtsam::I_3x3 * 1e-4 * 1e-4;
  p->biasAccOmegaInt = gtsam::I_6x6 * 1e-4 * 1e-4;

  // FOR EUROC MACHINE ROOM 
  // gtsam::Rot3 rot_IL(0, 0, 1,
  //                 0, -1, 0,
  //                 1, 0, 0);
  gtsam::Rot3 rot_IL = gtsam::Rot3::identity();
  gtsam::Point3 trans_IL(7.48903e-02, -1.84772e-02, -1.20209e-01);

  // FOR EUROC VICON ROOM 
  gtsam::Rot3 rot_IV(0.33638, -0.01749,  0.94156,
                 -0.02078, -0.99972, -0.01114,
                  0.94150, -0.01582, -0.33665);
  gtsam::Point3 trans_IV(0.06901, -0.02781, -0.12395);

  gtsam::Pose3 T_imu_mocap;
  ros::Subscriber extPosSub;
  if (strcmp(argv[1], "leica") == 0)
  {
    /// EUROC MACHINE HALL
    T_imu_mocap = gtsam::Pose3(rot_IL, trans_IL);
    extPosSub = nh.subscribe<geometry_msgs::PointStamped>("/leica/position", 100, leicaCallback);
  }
  else if (strcmp(argv[1], "vicon") == 0)
  {
    /// EUROC VICON ROOM
    T_imu_mocap = gtsam::Pose3(rot_IV, trans_IV);
    extPosSub = nh.subscribe<geometry_msgs::TransformStamped>("/vicon/firefly_sbx/firefly_sbx", 100, viconCallback);
  }
  else
  {
    ROS_WARN_STREAM("Dataset '" << argv[1] << "' not recognized. Only allowed with 'leica' or 'vicon' (case sensitive).");
    return 1;
  }

  p->body_P_sensor = T_imu_mocap.inverse();
  preint = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(p);

  ros::Subscriber imuSub = nh.subscribe<sensor_msgs::Imu>("/imu0", 100, imuCallback);
  ros::Subscriber imgSub = nh.subscribe<sensor_msgs::Image>("/cam0/image_raw", 10, imgCallback);

  std::cout << "Using backend type: '";
  switch (backend_type)
  {
  case BackendType::LM:
    std::cout << "Levenberg-Marquardt";
    break;
  case BackendType::ISAM:
    std::cout << "iSAM2";
    break;
  case BackendType::IFL:
    std::cout << "Incremental fixed lag";
    break;
  }
  std::cout << "'" << std::endl;
  std::cout << "INITIALIZED AND READY." << std::endl;

  ros::spin();
  return 0;
}
