## Realsense D435

如何将深度图像传输

- ros
  1. 机器人开机， 连接机器人WIFI， ssh远程登录
    WIFI名： unitree-xxxx...... (password: 00000000)
    `ssh unitree@192.168.123.12`
    密码:123
  2. 修改环境变量中ROS主机和ROS从机的IP地址
    主机地址：http://192.168.123.12:11311
    从机地址：ifconfig，192.168.123.xxx
    ```bash
      # for PC
      sudo gedit ~/.bashrc
      export ROS_MASTER_URI=http://192.168.123.12:11311  # A1 IP address
      export ROS_IP=192.168.123.226:11311	# PC IP address

      # for A1
      sudo gedit ~/.bashrc
      export ROS_MASTER_URI=http://192.168.123.12:11311
      export ROS_IP=192.168.123.12:11311
    ```
  3. launch摄像头节点，打开rqt可视化界面，rosbag录制视频
    ```bash
      # A1
      roslaunch realsense2_camera rs_camera.launch # (rostopic list检查话题是否成功建立)

      # client (PC)
      rosrun rqt_image_view rqt_image_view # 选择话题 /camera/color/image_raw
      rosbag record /camera/color/image_raw # 记录 RGB 图像
    ```
  4. 回放
    ```bash
      rosbag play xxx.bag
      rosrun rqt_image_view rqt_image_view
    ```
