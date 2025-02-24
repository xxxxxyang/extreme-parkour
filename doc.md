# 说明文档

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

得到的log目录结构为 `extreme-parkour/legged_gym/legged_gym/logs/proj_name/exptid/`

使用示例：
- 训练基础策略(with no camera)
```bash
python train.py --exptid baset0 --proj_name test --device cuda:0 --max_iterations 15000
```
推荐 10000-15000 次迭代 (10k-15k)
- 训练视觉策略
```bash
python train.py --exptid test-0 --device cuda:0 --resume --resumeid baset0 --delay --use_camera --max_iterations 5000
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
- 运行基础策略(with no camera)，使用网页渲染展示界面
```bash
python play.py --exptid baset0 --proj_name test --device cuda:0 --no_wandb --web
```
- 运行具有深度相机策略
```bash
python play.py --exptid test-0 --device cuda:0 --delay --use_camera --web
```

## 控制与命令方面

最后的策略是通过什么命令控制的，遥控器吗，遥控器的命令是位置命令还是线速度命令还是方向命令
在每一个 env 中，agent 的 command 都是持续该 episode 始终的吗，那与实际的操控是否有差别；如果有，通过什么样 randomization 可以 bridge 这种 gap
env 中的命令是一个四元组 [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]

## headless

在非无头模式下，即 headless=False 的情况下，gymapi.create_viewer 会创建出一个可视化窗口以观察仿真

## 实验平台

unitree a1/go1

## 地形方面

是每个 episode 后，该 env 都会改变 terrian 吗

## 关于时间

iteration step episode

每个 iteration 中执行 num_steps_per_env 个 step，即每个 env 在一次 iteration 中执行 1 个 step
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

噪声、光照、纹理、延迟?

1. 光照变化-强光过曝、低光乘性噪声：问题在于强/弱光影响区域
2. 材质/纹理相关噪声：高反射、折射表面，纹理噪声
3. 遮挡噪声

关键问题：调整区域、噪声比例、噪声频率、分布选择

## 视觉相机方面

深度相机 depth_camera 相关
###  legged_robot.py:
- self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), interpolation=torchvision.transforms.InterpolationMode.BICUBIC) 将深度相机获取的深度图像进行裁剪并进行插值以防止丢失过多的信息
- **为深度图像添加噪声**
- *深度缓冲区的作用*：保存观察到的深度图像，在reset等情况下更新；update_interval 的作用为降低深度图像更新的频率，降低计算开销，若 update_interval = 5 表示每 5 steps 更新一次 能否用 update_interval 模拟深度图像延迟

### legged_robot_config
- class depth_encoder: 设置在使用深度相机是训练的算法参数
- 


### on_policy_runner.py
- learn_vision 中为什么只进行 depth_actor 的参数更新，而不进行 encoder 的参数更新，好像也没有加载其预训练的模型吧；以及为什么 actions_teacher 不用 scandots；论文中应该是用的吧
- 在



**如何添加一个 RGBD 相机 rgb_camera**

*为什么启用相机后要修改 terrain 中的 y_range*



# doc
在 Isaac Gym 或类似仿真环境中，对二维张量的深度图像（`depth_image`）通过后处理注入噪声以实现域随机化（Domain Randomization），可以有效模拟真实世界中光照变化、材质差异、遮挡等因素对深度传感器的影响。以下是针对不同域随机化场景的噪声注入方法及代码实现：

---

### **1. 噪声注入的核心思路**
- **输入**：原始的深度图像（`depth_image`，形状为 `(H, W)`，单位为米或其他距离单位）。
- **输出**：添加噪声后的深度图像，模拟不同环境干扰。
- **关键步骤**：
  1. **深度值有效性检查**：处理无效值（如 `0` 或 `NaN`）。
  2. **分模块设计噪声**：针对光照、纹理、遮挡等场景设计独立的噪声模型。
  3. **参数随机化**：在每次调用时随机选择噪声类型、强度、位置等参数。

---

### **2. 分模块噪声模型设计**
以下为不同域随机化场景的噪声实现方法：

#### **(1) 光照相关噪声**
模拟光照对深度传感器的干扰（如过曝、低光噪声）：
```python
import numpy as np

def add_light_noise(depth_image, light_intensity_prob=0.3, max_intensity=5.0):
    """
    模拟光照变化导致的噪声：
    - 强光（过曝）：随机遮挡部分区域（深度置零）
    - 低光：添加乘性噪声（远距离噪声更大）
    """
    h, w = depth_image.shape
    noisy_depth = depth_image.copy()
    
    # 随机生成光照强度参数
    light_intensity = np.random.uniform(0, max_intensity)
    
    # 过曝噪声：按概率遮挡区域
    if np.random.rand() < light_intensity_prob:
        # 生成随机遮挡掩码（圆形或矩形）
        mask = np.zeros((h, w), dtype=bool)
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(10, 50)
        y, x = np.ogrid[:h, :w]
        mask_area = (x - cx)**2 + (y - cy)**2 <= radius**2
        noisy_depth[mask_area] = 0  # 深度值丢失
    
    # 低光噪声：乘性高斯噪声（噪声强度与深度成正比）
    noise_scale = np.random.uniform(0.01, 0.1)
    noise = noise_scale * noisy_depth * np.random.randn(h, w)
    noisy_depth = noisy_depth + noise
    
    return np.clip(noisy_depth, 0, None)  # 确保非负
```

#### **(2) 材质/纹理相关噪声**
模拟不同表面材质对深度测量的影响（如高反光、透明材质）：
```python
def add_material_noise(depth_image, reflectivity_prob=0.2, texture_scale=0.05):
    """
    模拟材质特性导致的噪声：
    - 高反射表面：随机生成错误深度值（如镜面反射导致的跳跃误差）
    - 纹理噪声：添加高频噪声模拟粗糙表面
    """
    h, w = depth_image.shape
    noisy_depth = depth_image.copy()
    
    # 高反光材质：随机位置生成跳跃误差
    if np.random.rand() < reflectivity_prob:
        num_spikes = np.random.randint(1, 5)
        for _ in range(num_spikes):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            spike_value = np.random.uniform(0, 10)  # 错误深度值
            noisy_depth[y, x] = spike_value
    
    # 纹理噪声：Perlin噪声或高频高斯噪声
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)
    texture_noise = np.sin(50 * X) * np.cos(30 * Y)  # 高频纹理
    noisy_depth += texture_scale * texture_noise
    
    return noisy_depth
```

#### **(3) 遮挡相关噪声**
模拟动态遮挡物（如飞溅的物体、临时障碍物）：
```python
def add_occlusion_noise(depth_image, max_occlusion_num=3):
    """
    模拟随机遮挡物：
    - 生成矩形或椭圆形的遮挡区域
    - 遮挡深度设为0或随机值
    """
    h, w = depth_image.shape
    noisy_depth = depth_image.copy()
    
    # 随机生成遮挡物数量和形状
    num_occlusions = np.random.randint(1, max_occlusion_num+1)
    for _ in range(num_occlusions):
        # 随机位置和大小
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        x2 = np.random.randint(x1, min(x1 + 50, w))
        y2 = np.random.randint(y1, min(y1 + 50, h))
        
        # 矩形或椭圆选择
        if np.random.rand() < 0.5:
            # 矩形遮挡
            noisy_depth[y1:y2, x1:x2] = 0
        else:
            # 椭圆遮挡
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
            y_grid, x_grid = np.ogrid[:h, :w]
            mask = ((x_grid - cx)/rx)**2 + ((y_grid - cy)/ry)**2 <= 1
            noisy_depth[mask] = 0
    
    return noisy_depth
```

#### **(4) 传感器故障噪声**
模拟传感器硬件噪声（如随机噪声、漂移）：
```python
def add_sensor_noise(depth_image, noise_type='gaussian'):
    """
    模拟传感器固有噪声：
    - 高斯噪声：普通测量误差
    - 脉冲噪声：随机椒盐噪声
    - 深度相关噪声：噪声强度随深度增加
    """
    h, w = depth_image.shape
    noisy_depth = depth_image.copy()
    
    # 高斯噪声
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 0.02, (h, w))  # 标准差0.02米
        noisy_depth += noise
    
    # 椒盐噪声（脉冲噪声）
    elif noise_type == 'salt_pepper':
        salt_pepper_ratio = 0.005
        num_salt = int(salt_pepper_ratio * h * w)
        coords = [np.random.randint(0, i-1, num_salt) for i in (h, w)]
        noisy_depth[coords[0], coords[1]] = np.random.uniform(0, 10, num_salt)  # 随机异常值
    
    # 深度相关噪声（远距离噪声更大）
    elif noise_type == 'depth_dependent':
        depth_scaled = noisy_depth / 10.0  # 假设最大深度10米
        noise = 0.1 * depth_scaled * np.random.randn(h, w)
        noisy_depth += noise
    
    return np.clip(noisy_depth, 0, None)
```

---

### **3. 综合域随机化流水线**
将上述噪声模型组合，按随机顺序和参数应用：
```python
def domain_randomization_pipeline(depth_image):
    """
    综合噪声注入流水线：
    - 随机选择噪声类型和参数
    - 按随机顺序叠加噪声
    """
    # 复制原始数据，避免原地修改
    noisy_depth = depth_image.copy()
    
    # 定义可选的噪声模块及参数
    noise_functions = [
        (add_light_noise, {'light_intensity_prob': 0.3}),
        (add_material_noise, {'reflectivity_prob': 0.2}),
        (add_occlusion_noise, {'max_occlusion_num': 3}),
        (add_sensor_noise, {'noise_type': 'gaussian'})
    ]
    
    # 打乱噪声应用顺序
    np.random.shuffle(noise_functions)
    
    # 依次应用噪声（可随机跳过部分噪声）
    for func, kwargs in noise_functions:
        if np.random.rand() < 0.7:  # 70%概率应用该噪声
            noisy_depth = func(noisy_depth, **kwargs)
    
    return noisy_depth
```

---

### **4. 使用示例**
在 Isaac Gym 中获取深度图后应用噪声：
```python
import isaacgym
from isaacgym import gymapi

# 初始化环境并获取深度图像
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
env = gym.create_env(sim_params, gymapi.Vec3(0, 0, 0), 1, 1)
camera_handle = gym.create_camera(env, gymapi.CameraProperties())
depth_image = gym.get_camera_image(env, camera_handle, gymapi.IMAGE_DEPTH)

# 应用域随机化
noisy_depth = domain_randomization_pipeline(depth_image)

# 可视化对比原始和带噪声的深度图
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(depth_image, cmap='jet')
plt.title('Original Depth')
plt.subplot(1, 2, 2)
plt.imshow(noisy_depth, cmap='jet')
plt.title('Noisy Depth')
plt.show()
```

---

### **5. 高级优化技巧**
- **GPU加速**：使用 `cupy` 或 `PyTorch` 将噪声计算迁移到 GPU。
- **无效值保护**：在噪声注入前标记无效区域（如 `depth_image == 0`），避免破坏原始信息。
- **物理合理性**：
  - 对遮挡噪声，可根据场景动态生成遮挡物的 3D 模型（而非后处理）。
  - 对反射噪声，结合射线追踪计算二次反射路径。
- **参数分布调整**：使用概率分布（如 Beta 分布）控制噪声强度，而非均匀分布。

---

### **6. 效果示例**
| **噪声类型**       | **输入深度图**                     | **输出深度图**                     |
|--------------------|----------------------------------|----------------------------------|
| 光照噪声（过曝）    | ![原始深度图](img/original.png)  | ![过曝噪声](img/light_noise.png) |
| 材质噪声（高反光）  | ![原始深度图](img/original.png)  | ![材质噪声](img/material_noise.png) |
| 遮挡噪声           | ![原始深度图](img/original.png)  | ![遮挡噪声](img/occlusion.png)   |

---

通过灵活组合上述方法，可以在仿真中生成高度多样化的深度数据，有效提升模型在真实场景中的鲁棒性。