# Corruption Benchmark

English | [简体中文](README_CN.md)

## Online Usage with MMDetection3D-Based Frameworks


#### Manual Integration


1. Copy the following files and directory into `/mmdet3d/datasets/pipelines/`:

   - [Camera_corruptions.py](./mmdet3d/datasets/pipelines/Camera_corruptions.py)
   - [LiDAR_corruptions.py](./mmdet3d/datasets/pipelines/LiDAR_corruptions.py)
   - [util/](./mmdet3d/datasets/pipelines/util/)

2. Merge the `CorruptionMethods` class, including its `@PIPELINES.register_module()` decorator, from the [test_time_aug.py](./mmdet3d/datasets/pipelines/test_time_aug.py) into the target's `mmdet3d/datasets/pipelines/test_time_aug.py`. Also merge the imports used by this class and the three helper functions: `format_list_float_06`, `load_points`, and `remove_close`.

3. Add the following import to the target's `mmdet3d/datasets/pipelines/__init__.py`:

   ```python
   from .test_time_aug import CorruptionMethods
   ```

### 2. Update the Test Pipeline Configuration

In the configuration file's `test_pipeline`, insert the following entry after loading images, point clouds, and 3D annotations, and before `ImageAug3D`:

```yaml
- type: CorruptionMethods
  corruption_severity_dict:
    echo_depletion: 3
```

This configuration enables the `echo_depletion` corruption for a sample with severity `3`.
