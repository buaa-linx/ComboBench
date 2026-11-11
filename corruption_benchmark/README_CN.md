# Corruption Benchmark

[English](README.md) | 简体中文

## 在线使用：基于 MMDetection3D 的框架

#### 手动集成

1. 将以下文件和目录复制到 `<BEVFUSION_ROOT>/mmdet3d/datasets/pipelines/`：

   - [Camera_corruptions.py](./mmdet3d/datasets/pipelines/Camera_corruptions.py)
   - [LiDAR_corruptions.py](./mmdet3d/datasets/pipelines/LiDAR_corruptions.py)
   - [util/](./mmdet3d/datasets/pipelines/util/)

2. 将源文件 [test_time_aug.py](./mmdet3d/datasets/pipelines/test_time_aug.py) 中带有 `@PIPELINES.register_module()` 的 `CorruptionMethods` 类合并到目标的 `mmdet3d/datasets/pipelines/test_time_aug.py` 中。同时合并该类使用的导入，以及 `format_list_float_06`、`load_points`、`remove_close` 三个辅助函数。

3. 在目标的 `mmdet3d/datasets/pipelines/__init__.py` 中添加以下导入；如果已经存在，则不必重复添加：

   ```python
   from .test_time_aug import CorruptionMethods
   ```

### 2. 修改测试管线配置

在配置文件的 `test_pipeline` 中，图像、点云和 3D 标注加载完成后、`ImageAug3D` 之前，插入以下配置项：

```yaml
- type: CorruptionMethods
  corruption_severity_dict:
    echo_depletion: 3
```

该配置为样本启用 `echo_depletion` 异常，强度为 `3`。




