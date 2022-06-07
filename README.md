# mpc_tracer-for-ROS
mpc_tracker for ROS, wrapped as a local_planner plugin.

* A package "api2python" follows the format of local_planner in ROS navigation stack. It calls the mpc tracer by SERVICE. 
* Another package wraps Model Predictive Control (MPC) by python and cvxpy. It returns the action for the robot. 
