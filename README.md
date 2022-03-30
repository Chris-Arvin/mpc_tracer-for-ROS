# mpc_tracer-for-ROS
mpc_tracker for ROS, wrapped as a local_planner plugin.

* api2python功能包的c++格式按照ROS的navigation stack的local_planner的plugin格式写的，通过service机制调用mpc_tracker【此功能包也可以用来与其他文件配合使用，来使得用户可以用python来写自己的local planner算法】
* mpc_tracer功能包python文件，接收api2python的消息
