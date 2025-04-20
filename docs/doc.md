# 说明文档

在多 gpu 环境下，若训练带有 camera 的策略是遇到 
```bash
[Error] [carb.gym.plugin] cudaExternamMemoryGetMappedBuffer failed on segmentationImage buffer with error 101
```
可以设置当前可见 gpu 仅为一个，例如：(根据你的 nvidia-smi 结果确定)
```bash
export CUDA_VISIBLE_DEVICES=1
```

## 参数

必要参数：
1. **train.py**:
--proj_name     当前项目目录名
--exptid        当前训练的编号 str(注意：不要使用具有前六个字符相同的编号)
--use_camera    使用深度相机
--delay         使用动作延迟    在 iteration > 8k 是添加
--load_run      已弃用，默认为-1，不必有其他操作
--checkpoint    默认为-1，即从 log/proj_name/resumeid/ 下获取最后一次训练的 checkpoint 模型(.pt)
--resume        启用加载预训练模型
--resumeid      预训练模型id，选取想要加载的exptid即可 str
--device        训练设备，默认为 cuda:0
--max_iterations    最大迭代次数
--no_drand      不使用动力学 domain_rand
--noise         添加相机噪声

得到的log目录结构为 `extreme-parkour/legged_gym/legged_gym/logs/proj_name/exptid/`

使用示例：
- 训练基础策略(with no camera)
```bash
python train.py --exptid baset0 --proj_name test --device cuda:0 --max_iterations 15000
```
推荐 10000-15000 次迭代 (10k-15k)
- 训练视觉策略
```bash
python train.py --exptid test-0 --device cuda:0 --resume --resumeid baset0 --delay --use_camera --max_iterations 5000 --headless (--noise)
```
推荐 5000 次迭代 (5k)

2. **play.py**:
--web           使用web浏览器查看和控制模拟环境中的相机视图
--no_wandb      不使用wandb记录
--proj_name     需要使用的策略的项目目录名
--exptid        需要使用的策略项目名
--use_camera    使用有深度相机的策略
--delay         使用动作延迟

使用示例：
- 运行基础策略(with no camera)，使用网页 (web) 渲染展示界面
```bash
python play.py --exptid baset0 --proj_name test --device cuda:0 --no_wandb --web
```
- 运行具有深度相机策略
```bash
python play.py --exptid test-0 --device cuda:0 --delay --use_camera --web --no_wandb (--noise)
```
- 运行关于最终策略的评估
```bash
python evaluate.py --exptid test-0 --device cuda:0 --delay --use_camera --no_wandb (--noise)
```

*关于 flask 服务器*：
当 host 设置为 `0.0.0.0` ，可以通过 `http://service-ip:port/` 访问根URL

## 控制与命令方面

最后的策略是通过什么命令控制的，遥控器吗，遥控器的命令是位置命令还是线速度命令还是方向命令
在每一个 env 中，agent 的 command 都是持续该 episode 始终的吗，那与实际的操控是否有差别；如果有，通过什么样 randomization 可以 bridge 这种 gap
env 中的命令是一个四元组 [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]

**仿真中如何控制命令的？**

## headless

在非无头模式下，即 headless=False 的情况下，gymapi.create_viewer 会创建出一个可视化窗口以观察仿真


## 实验平台

unitree a1/go1

## 相机方面

深度相机 depth_camera 相关

###  legged_robot.py:
- self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), interpolation=torchvision.transforms.InterpolationMode.BICUBIC) 将深度相机获取的深度图像进行裁剪并进行插值以防止丢失过多的信息
- **为深度图像添加噪声**
- *深度缓冲区的作用*：保存观察到的深度图像，在reset等情况下更新；update_interval 的作用为降低深度图像更新的频率，降低计算开销，若 update_interval = 5 表示每 5 steps 更新一次 能否用 update_interval 模拟深度图像延迟

去噪方法：
需要研究现在的图像帧率

### legged_robot_config
- class depth_encoder: 设置在使用深度相机是训练的算法参数
- 


### on_policy_runner.py
- learn_vision 中为什么只进行 depth_actor 的参数更新，而不进行 encoder 的参数更新，好像也没有加载其预训练的模型吧；以及为什么 actions_teacher 不用 scandots；论文中应该是用的吧
- 在



**为什么启用相机后要修改 terrain 中的 y_range**
**如何进行深度相机的域随机化**
**为什么启用相机后要修改 terrain 中的 y_range？**

## 地形方面

是每个 episode 后，该 env 都会改变 terrian 吗


## 关于时间

iteration step episode

控制步长和仿真步长：
控制步长在本仓库中可理解为策略网络更新 `action` 的步长，与 `iteration` 同频，`step` 的频率是 `iteration` 的 `num_steps_per_env` 倍
仿真步长(self.cfg.sim.dt=0.005)可理解为应用一次 `action` 的步长，在一次 `step` 中会更新 `decimation` 次，使得仿真结果更加准确和稳定

回合最大长度 `max_episode_length` 由 `max_episode_length_s / dt` 决定

**learn_RL**:
每个 `iteration` 中执行 `num_steps_per_env` 个 step，即每个 `env` 在一次 `iteration` 中执行 `num_steps_per_env` 个 `step`；其中每个 `step` 持续时间为 `dt`(self.cfg.sim.dt x decimation = 0.005x4 = 0.02s) 
(注意，self.dt 是 step 的实际仿真步长时间，而 self.cfg.sim.dt 与 self.sim_params.dt 相同，表示原始的仿真步长，没有与 decimation 计算)

**learn_vision**
每个 `iteration` 中执行 `update_interval*num_steps_per_env` 个 `step`，每个 `step` 持续时间仍为 `dt`
每 `update_interval` 个 `step` 更新一次 `update_depth_buffer`，即**深度图像的帧率为 `dt x update_interval = 1/0.02x5 = 10(帧/s)`**

## 本体感觉

仿真中观察obs的维度：

**obs包含了当前动作，包含之前的几个动作是否会有帮助?**

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
5. **动作延迟相关参数**  动作延迟的大小通过 dt 控制
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

光照条件的仿真是重点

噪声、光照、纹理、延迟?

1. 光照变化-强光过曝、低光乘性噪声：问题在于强/弱光影响区域
2. 材质/纹理相关噪声：高反射、折射表面，纹理噪声
3. 遮挡噪声

关键问题：调整区域、噪声比例、噪声频率、分布选择

```python
# 深度图像大小：[]
# depth_camera
        randomize_camera = False    # randomize camera position, angle and fov
        camera_pos_range = [[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]] # [m]
        camera_angle_range = [-5, 5] # [deg]
        camera_fov_range = [60, 120] # [deg]

        randomize_depth_noise = False   # randomize camera noise

        light_intensity_prob = 0.2  # probability of changing light intensity
        max_intensity  = 5.0   # max light intensity
        light_rand_type = "uniform"  # 'uniform' or 'normal'

        reflectivity_prob = 0.2   # probability of changing reflectivity
        texture_scale = 0.05  # scale of the texture

        max_occlusion_num = 3 # max number of occlusions
        max_occ_width = 20    # max width of occlusions
        max_occ_height = 20   # max height of occlusions

        noise_type = "gaussian"   # noise type 'gaussian' or 'salt_pepper' or 'dedepth_dependent'
```


## 奖励方面

mean_reward_task 是关键，即排除探索奖励的部分，反应出该 agent 在当前任务中的执行能力



