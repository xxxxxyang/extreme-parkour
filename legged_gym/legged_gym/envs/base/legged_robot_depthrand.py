from legged_gym.envs.base.legged_robot import LeggedRobot, LeggedRobotCfg, euler_from_quaternion
from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch
import numpy as np
import torch.nn.functional as F
import random
import torch
import cv2

class DepthRand(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.max_depth = 0
        self.min_depth = 0
        self.mean_depth = 0
        self.std_depth = 0
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


    def _attach_camera(self, i, env_handle, actor_handle):
        """
        Attach a camera to the robot
        Args:
            i (int): camera index
            env_handle (gymapi.Env): environment handle
            actor_handle (gymapi.Actor): actor handle

        TODO: add the randomization of camera position and angle
        """
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov if (not self.cfg.domain_rand.randomize_camera) else np.random.uniform(self.cfg.domain_rand.camera_fov_range[0], self.cfg.domain_rand.camera_fov_range[1])
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            # camera_position = np.copy(config.position)
            # camera_angle = np.random.uniform(config.angle[0], config.angle[1])

            # domain_rand of camera position and angle
            if self.cfg.domain_rand.randomize_camera:
                # TODO add randomization of camera pos and angle
                camera_position = np.copy([np.random.uniform(config.position[0]*(1-self.cfg.domain_rand.camera_pos_range), config.position[0]*(1+self.cfg.domain_rand.camera_pos_range)),
                                            np.random.uniform(config.position[1]*(1-self.cfg.domain_rand.camera_pos_range), config.position[1]*(1+self.cfg.domain_rand.camera_pos_range)),
                                            np.random.uniform(config.position[2]*(1-self.cfg.domain_rand.camera_pos_range), config.position[2]*(1+self.cfg.domain_rand.camera_pos_range))])
                camera_angle = np.random.uniform(self.cfg.domain_rand.camera_angle_range[0], self.cfg.domain_rand.camera_angle_range[1])
            else:
                camera_position = np.copy(config.position)
                camera_angle = np.random.uniform(config.angle[0], config.angle[1])

            
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)    

    ########## Steps to simulate depth image noise ##########
    def _add_depth_contour(self, depth_images):
        """ Simulate the contour noise of the depth image.
        depth_images: [num_envs, H, W]
        """
        depth_images = depth_images.unsqueeze(1) # [num_envs, 1, H, W] add channel dim
        self.contour_detection_kernel = torch.zeros(
                (8, 1, 3, 3),
                dtype= torch.float32,
                device= self.device,
            )
            # emperical values to be more sensitive to vertical edges
        self.contour_detection_kernel[0, :, 1, 1] = 0.5
        self.contour_detection_kernel[0, :, 0, 0] = -0.5
        self.contour_detection_kernel[1, :, 1, 1] = 0.1
        self.contour_detection_kernel[1, :, 0, 1] = -0.1
        self.contour_detection_kernel[2, :, 1, 1] = 0.5
        self.contour_detection_kernel[2, :, 0, 2] = -0.5
        self.contour_detection_kernel[3, :, 1, 1] = 1.2
        self.contour_detection_kernel[3, :, 1, 0] = -1.2
        self.contour_detection_kernel[4, :, 1, 1] = 1.2
        self.contour_detection_kernel[4, :, 1, 2] = -1.2
        self.contour_detection_kernel[5, :, 1, 1] = 0.5
        self.contour_detection_kernel[5, :, 2, 0] = -0.5
        self.contour_detection_kernel[6, :, 1, 1] = 0.1
        self.contour_detection_kernel[6, :, 2, 1] = -0.1
        self.contour_detection_kernel[7, :, 1, 1] = 0.5
        self.contour_detection_kernel[7, :, 2, 2] = -0.5
        edge_mask = torch.abs(F.conv2d(depth_images, self.contour_detection_kernel, padding= 1)).max(dim= -3, keepdim= True)[0]
        mask =  F.max_pool2d(
            edge_mask,
            kernel_size = 3,
            stride = 1,
            padding = int(3 / 2),
        ) > self.cfg.domain_rand.contour_threshold
        # dim weight
        blur_kernel_size = 3
        blur_weights = torch.ones((1, 1, blur_kernel_size, blur_kernel_size), device=self.device) / (blur_kernel_size ** 2)
        # blur the depth image
        blurred_depth = F.conv2d(depth_images, blur_weights, padding=blur_kernel_size // 2)
        depth_images[mask] = blurred_depth[mask]

        return depth_images.squeeze(1) # [num_envs, H, W] remove channel dim

    @torch.no_grad()
    def form_artifacts(self,
            H, W, # image resolution
            tops, bottoms, # artifacts positions (in pixel) shape (n_,)
            lefts, rights,
        ):
        """ Paste an artifact to the depth image.
        NOTE: Using the paradigm of spatial transformer network to build the artifacts of the
        entire depth image.
        """
        batch_size = tops.shape[0]
        tops, bottoms = tops[:, None, None], bottoms[:, None, None]
        lefts, rights = lefts[:, None, None], rights[:, None, None]

        # build the source patch
        source_patch = torch.zeros((batch_size, 25, 25), device= self.device)
        source_patch[:, 1:24, 1:24] = 1.

        # build the grid
        grid = torch.zeros((batch_size, H, W, 2), device= self.device)
        grid[..., 0] = torch.linspace(-1, 1, W, device= self.device).view(1, 1, W)
        grid[..., 1] = torch.linspace(-1, 1, H, device= self.device).view(1, H, 1)
        grid[..., 0] = (grid[..., 0] * W + W - rights - lefts) / (rights - lefts)
        grid[..., 1] = (grid[..., 1] * H + H - bottoms - tops) / (bottoms - tops)

        # sample using the grid and form the artifacts for the entire depth image
        artifacts = torch.clip(
            F.grid_sample(
                source_patch.unsqueeze(1),
                grid,
                mode= "bilinear",
                padding_mode= "zeros",
                align_corners= False,
            ).sum(dim= 0).view(H, W),
            0, 1,
        )

        return artifacts

    def _add_depth_artifacts(self, depth_images,
            artifacts_prob,
            artifacts_height_mean_std,
            artifacts_width_mean_std,
        ):
        """ Simulate artifacts from stereo depth camera. In the final artifacts_mask, where there
        should be an artifacts, the mask is 1.
        depth_images: [num_envs, H, W]
        """
        N, H, W = depth_images.shape
        def _clip(x, dim):
            return torch.clip(x, 0., (H, W)[dim])

        # random patched artifacts
        artifacts_mask = torch_rand_float(
            0., 1.,
            (N, H * W),
            device= self.device,
        ).view(N, H, W) < artifacts_prob
        artifacts_mask = artifacts_mask & (depth_images > 0.)
        artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32) # (n_, 3) n_ <= N * H * W
        artifcats_size = (
            torch.clip(
                artifacts_height_mean_std[0] + torch.randn(
                    (artifacts_coord.shape[0],),
                    device= self.device,
                ) * artifacts_height_mean_std[1],
                0., H,
            ),
            torch.clip(
                artifacts_width_mean_std[0] + torch.randn(
                    (artifacts_coord.shape[0],),
                    device= self.device,
                ) * artifacts_width_mean_std[1],
                0., W,
            ),
        ) # (n_,), (n_,)
        artifacts_top_left = (
            _clip(artifacts_coord[:, 1] - artifcats_size[0] / 2, 0),
            _clip(artifacts_coord[:, 2] - artifcats_size[1] / 2, 1),
        )
        artifacts_bottom_right = (
            _clip(artifacts_coord[:, 1] + artifcats_size[0] / 2, 0),
            _clip(artifacts_coord[:, 2] + artifcats_size[1] / 2, 1),
        )
        for i in range(N):
            # NOTE: make sure the artifacts points are as few as possible
            artifacts_mask = self.form_artifacts(
                H, W,
                artifacts_top_left[0][artifacts_coord[:, 0] == i],
                artifacts_bottom_right[0][artifacts_coord[:, 0] == i],
                artifacts_top_left[1][artifacts_coord[:, 0] == i],
                artifacts_bottom_right[1][artifacts_coord[:, 0] == i],
            )
            depth_images[i] *= (1 - artifacts_mask)

        return depth_images
    
    def _recognize_top_down_too_close(self, too_close_mask):
        """ Based on real D435i image pattern, there are two situations when pixels are too close
        Whether there is too-close pixels all the way across the image vertically.
        """
        # vertical_all_too_close = too_close_mask.all(dim= 2, keepdim= True)
        vertical_too_close = too_close_mask.sum(dim= -2, keepdim= True) > (too_close_mask.shape[-2] * 0.6)
        return vertical_too_close
    
    def _add_depth_stereo(self, depth_images):
        """ Simulate the noise from the depth limit of the stereo camera. """
        N, H, W = depth_images.shape
        far_mask = depth_images > self.cfg.domain_rand.stereo_far_distance
        too_close_mask = depth_images < self.cfg.domain_rand.stereo_min_distance
        near_mask = (~far_mask) & (~too_close_mask)

        # add noise to the far points
        far_noise = torch_rand_float(
            0., self.cfg.domain_rand.stereo_far_noise_range,
            (N, H * W),
            device= self.device,
        ).view(N, H, W)
        far_noise = far_noise * far_mask
        depth_images += far_noise

        # add noise to the near points
        near_noise = torch_rand_float(
            0., self.cfg.domain_rand.stereo_near_noise_range,
            (N, H * W),
            device= self.device,
        ).view(N, H, W)
        near_noise = near_noise * near_mask
        depth_images += near_noise

        # add artifacts to the too close points
        vertical_block_mask = self._recognize_top_down_too_close(too_close_mask)
        full_block_mask = vertical_block_mask & too_close_mask
        half_block_mask = (~vertical_block_mask) & too_close_mask
        # add artifacts where vertical pixels are all too close
        for pixel_value in random.sample(
                self.cfg.domain_rand.stereo_full_block_values,
                len(self.cfg.domain_rand.stereo_full_block_values),
            ):
            artifacts_buffer = torch.ones_like(depth_images)
            artifacts_buffer = self._add_depth_artifacts(artifacts_buffer,
                self.cfg.domain_rand.stereo_full_block_artifacts_prob,
                self.cfg.domain_rand.stereo_full_block_height_mean_std,
                self.cfg.domain_rand.stereo_full_block_width_mean_std,
            )
            depth_images[full_block_mask] = ((1 - artifacts_buffer) * pixel_value)[full_block_mask]
        # add artifacts where not all the same vertical pixels are too close
        half_block_spark = torch_rand_float(
            0., 1.,
            (N, H * W),
            device= self.device,
        ).view(N, H, W) < self.cfg.domain_rand.stereo_half_block_spark_prob
        depth_images[half_block_mask] = (half_block_spark.to(torch.float32) * self.cfg.domain_rand.stereo_half_block_value)[half_block_mask]

        return depth_images
    
    def _recognize_top_down_seeing_sky(self, too_far_mask):
        N, H, W = too_far_mask.shape
        # whether there is too-far pixels with all pixels above it too-far
        num_too_far_above = too_far_mask.cumsum(dim= -2)
        all_too_far_above_threshold = torch.arange(H, device= self.device).view(1, H, 1)
        all_too_far_above = num_too_far_above > all_too_far_above_threshold # (N, H, W) mask
        return all_too_far_above
    
    def _add_sky_artifacts(self, depth_images):
        """ Incase something like ceiling pattern or stereo failure happens. """
        N, H, W = depth_images.shape
        
        possible_to_sky_mask = depth_images > self.cfg.domain_rand.sky_artifacts_far_distance
        to_sky_mask = self._recognize_top_down_seeing_sky(possible_to_sky_mask)
        isinf_mask = depth_images.isinf()
        
        # add artifacts to the regions where they are seemingly pointing to sky
        for pixel_value in random.sample(
                self.cfg.domain_rand.sky_artifacts_values,
                len(self.cfg.domain_rand.sky_artifacts_values),
            ):
            artifacts_buffer = torch.ones_like(depth_images)
            artifacts_buffer = self._add_depth_artifacts(artifacts_buffer,
                self.cfg.domain_rand.sky_artifacts_prob,
                self.cfg.domain_rand.sky_artifacts_height_mean_std,
                self.cfg.domain_rand.sky_artifacts_width_mean_std,
            )
            depth_images[to_sky_mask & (~isinf_mask)] *= artifacts_buffer[to_sky_mask & (~isinf_mask)]
            depth_images[to_sky_mask & isinf_mask & (artifacts_buffer < 1)] = 0.
            depth_images[to_sky_mask] += ((1 - artifacts_buffer) * pixel_value)[to_sky_mask]
            pass
        
        return depth_images
    
    def rand_sensor_noise(self, depth_image):
        """
        Simulate sensor inherent noise:
        - Gaussian noise: typical measurement errors
        - Impulse noise: random salt-and-pepper noise
        - Depth-dependent noise: noise intensity increases with depth
        """
        noise_type = self.cfg.domain_rand.noise_type

        h, w = depth_image.shape
        noisy_depth = depth_image.clone()
        inf_mask = noisy_depth == float('-inf')
        noisy_depth[inf_mask] = 0

        # Randomly generate the noise intensity (with low intensity)

        if noise_type == 'gaussian':
            noise = torch.randn(h, w, device=self.device) * 0.1 * noisy_depth # TODO: Standard deviation of 10% of the depth
            noisy_depth += noise

        elif noise_type == 'salt_pepper':
            salt_pepper_ratio = 0.005
            num_salt = int(salt_pepper_ratio * h * w)
            coords = [torch.randint(0, i, (num_salt,), device=self.device) for i in [h, w]]
            noisy_depth[coords[0], coords[1]] = torch.rand(num_salt, device=depth_image.device) * 10  # Random outliers

        elif noise_type == 'depth_dependent':
            depth_scaled = noisy_depth / 10.0  # Assume maximum depth is 10 meters
            noise = 0.1 * depth_scaled * torch.randn(h, w, device=depth_image.device)
            noisy_depth -= noise

        noisy_depth[inf_mask] = float('-inf')
        return torch.clip(noisy_depth, None, 0) # depth should be negative

    ##### Depth Image Processing #####
    def add_depth_domain_rand_pipeline(self, depth_image):
        """
        Add depth domain randomization to the depth image
        Args:
            depth_image (torch.Tensor): depth image of shape (H, W)
        Returns:
            depth_image (torch.Tensor): depth image with domain randomization
        """
        # clip inf
        inf_mask = depth_image == float('-inf')
        depth_image[inf_mask] = -100
        # add contour
        depth_image = self._add_depth_contour(depth_image)
        # add stereo noise
        depth_image = self._add_depth_stereo(depth_image)
        # add sky artifacts
        depth_image = self._add_sky_artifacts(depth_image)
        # add sensor noise
        self.rand_sensor_noise(depth_image)

        # replace -100 with -inf
        depth_image[inf_mask] = float('-inf')
        return depth_image

    def normalize_depth_image(self, depth_image):
        """
        original depth image is in the range [0, 1]
        Normalize the depth image to be in the range [-0.5, 0.5]
        """
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def crop_depth_image(self, depth_image):
        """
        input: depth_image: [num_envs, origin_H, origin_W]
        output: [num_envs, H, W]
        """
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:, :-2, 4:-4]
    
    def process_depth_image(self, depth_image):
        """
        Process a batch of depth images, batchsize = num_envs
        depth_image: [num_envs, H, W]
        """
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        # add noise
        # depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        if self.cfg.domain_rand.randomize_depth_noise:
            depth_image = self.add_depth_domain_rand_pipeline(depth_image)
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    # get the depth image and process
    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        depth_image = torch.zeros((self.num_envs, self.cfg.depth.original[1], self.cfg.depth.original[0]), device=self.device)
        for i in range(self.num_envs):
            # get the depth image from the camera(negative depth value)
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image_ = gymtorch.wrap_tensor(depth_image_).to(self.device)
            self.max_depth = torch.abs(depth_image_[depth_image_ != float('-inf')]).max() if torch.abs(depth_image_[depth_image_ != float('-inf')]).max() > self.max_depth else self.max_depth
            self.min_depth = torch.abs(depth_image_[depth_image_ != float('-inf')]).min() if torch.abs(depth_image_[depth_image_ != float('-inf')]).min() < self.min_depth else self.min_depth
            self.mean_depth = torch.abs(depth_image_[depth_image_ != float('-inf')]).mean()
            self.std_depth = torch.abs(depth_image_[depth_image_ != float('-inf')]).std()
            # convert the depth image to the original size
            depth_image[i] = depth_image_.view(self.cfg.depth.original[1], self.cfg.depth.original[0])

        # process batch of depth images (N, H, W)
        # torch.set_printoptions(threshold=float('inf'))        
        depth_image = self.process_depth_image(depth_image)
        # torch.set_printoptions(profile="default")

        init_flag = self.episode_length_buf <= 1
        init_index = torch.nonzero(init_flag, as_tuple=True)    # (num_envs, )
        not_init_index = torch.nonzero(~init_flag, as_tuple=True) # (num_envs, )
        self.depth_buffer[init_index] = depth_image[init_index].unsqueeze(1).repeat(1, self.cfg.depth.buffer_len, 1, 1)
        self.depth_buffer[not_init_index] = torch.cat([self.depth_buffer[not_init_index, 1:], 
                                                       depth_image[not_init_index].unsqueeze(1)], dim=1)

        self.gym.end_access_image_tensors(self.sim)

    ######### Physics Step #########
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        
        # self._update_jump_schedule()
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self._draw_height_samples()
            self._draw_goals()
            self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
