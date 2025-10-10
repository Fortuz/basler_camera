# external_camera

Tasklist / Requirement
- ROS2 pkg
- Docker container setup
- The package should provide a data structure including: robot position and orientaion, human owner position orientation, human stranger position orientation, toy position and a bool variable for each object is it tracked or not. If on object is not tracked keep the previously observed position and orientation.
- Need to handle a Basler camara (connection, disconnection issues)
- Clear Readme including: setup instructions on ubuntu without docker and on windows with docker, bring up commands on how to start the actual code, some user manual part: what does the code do and how it is implemented
- Need to calibrate the camera to be alligned with the room
- Maybe an additional metadata servic would be required, to help the robot syncronize the camera coordinate system
