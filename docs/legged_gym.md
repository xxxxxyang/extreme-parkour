## 深度图像处理

深度图像的帧率为 `dt x update_interval = 1/0.02x5 = 10(帧/s)`；每经过 `update_interval` 个 `step` 更新一次 `depth_buffer`

depth_buffer 的类型为：[envs, buffer_len, height, width] 的四维张量