# External Camera Project

This project focuses on using an external "Basler" camera to track the mecanumbot and different types of Humans may it be owner or a regular human and a toy, across a inclosded space
The project is a 4 node project where we run the camera node , calibrate it , run the tracking node and use the yolo node for the full product

# Usage

At first we need to source our workspace and source ros
We build the package : colcon build --packages-select basler_camera
Run the camera node : ros2 run basler_camera camera_node  
After that we can calibrate the camera using chessboard calibration running : ros2 run basler_camera chessboard_calibration_node
To actually open the gui and track we use : ros2 run basler_camera yolo_node which runs a calibrated version of the camera, it detects five different classes
Mecanumbot, Mecanumbothead (used for determining mecanumbot direction), HumanOwner, HumanNormal, TennisBall. Giving their coordinates in in realtime and giving a bool value based on if it's being tracked or not, these values are also displayed in realtime in the terminal.
