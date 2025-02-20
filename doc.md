# 说明文档

## 实验平台

unitree a1/go1

## 相机方面

深度相机

如何添加一个 RGBD 相机

为什么启用相机后要修改 terrain 中的 y_range

## 地形方面

## 关于时间

iteration step episode

每个 iteration 中执行 num_steps_per_env 个 step
其中每个 step 持续时间为 dt ( actor仿真帧率为 1/dt Hz，策略迭代帧率为 1/ (dt*decimation) )

## 本体感觉

仿真中观察obs的维度：

**obs包含了当前动作，包含之前的几个动作是否会有帮助?**

实际：

## dynamic_rand

要想关掉所有 domain_rand 首先要确保不被 args 覆盖，其次是要确保没有 domain_rand 的时候要可以正常运行，即有默认值

1. 摩擦相关参数
    randomize_friction: 是否随机化摩擦系数。
    friction_range: 摩擦系数的范围。
2. 质量和质心相关参数
    randomize_base_mass: 是否随机化机器人的质量。
    added_mass_range: 质量增加的范围。
    randomize_base_com: 是否随机化机器人的质心位置。
    added_com_range: 质心位置增加的范围。
3. 推力相关参数
    push_robots: 是否在模拟中推机器人。
    push_interval_s: 推机器人的时间间隔（秒）。
    max_push_vel_xy: 推力的最大速度（在 x 和 y 方向）。
4. 电机相关参数
    randomize_motor: 是否随机化电机的强度。
    motor_strength_range: 电机强度的范围。
5. **动作延迟相关参数** ？什么情况下要加动作延迟 可以通过参数控制是否添加动作延迟
    delay_update_global_steps: 全局步数更新的延迟。
    action_delay: 是否启用动作延迟。
    action_curr_step: 当前动作步数。
    action_curr_step_scratch: 当前动作步数的初始值。
    action_delay_view: 动作延迟视图。
    action_buf_len: 动作缓冲区长度。
6. 其他参数 在LeggedRobotCfg的 env 子类中
    randomize_start_pos: 是否随机化初始位置。
    randomize_start_vel: 是否随机化初始速度。
    randomize_start_yaw: 是否随机化初始偏航角。
    rand_yaw_range: 初始偏航角的范围。
    randomize_start_y: 是否随机化初始 y 位置。
    rand_y_range: 初始 y 位置的范围。
    randomize_start_pitch: 是否随机化初始俯仰角。
    rand_pitch_range: 初始俯仰角的范围

## camera domain_rand


