# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).


## [Unreleased]


## [1.2.0] - denoise
### Added
- depth_temporal 文件，添加了使用 CNN 处理连续帧的逻辑
- 添加训练、测试中对连续帧图像处理的兼容

### Changed
- 修改了噪声域随机化的项目，只使用图像的噪声，暂无相机位置的随机化


## [1.1.0] - main
### Added
- 添加了深度图像的噪声域随机化
