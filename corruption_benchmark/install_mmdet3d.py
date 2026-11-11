#!/usr/bin/env python3
"""Install corruption overlay into a compatible BEVFusion/MMDetection3D checkout.

Only source files under pipelines/ are copied; existing files are backed up.
Import registry declaration is appended, not wholesale replaced.
"""
import argparse
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('--target', type=Path, required=True, help='Root folder of the matching BEVFusion fork')
a = p.parse_args()
source = Path(__file__).resolve().parent / 'mmdet3d_overlay' / 'mmdet3d' / 'datasets' / 'pipelines'
target = a.target.resolve() / 'mmdet3d' / 'datasets' / 'pipelines'
init = target / '__init__.py'
if not init.exists() or not (a.target / 'mmdet3d' / 'models').exists():
    p.error('Target must be a compatible BEVFusion/MMDetection3D project root')
for name in ('Camera_corruptions.py','LiDAR_corruptions.py','test_time_aug.py'):
    dest=target/name
    if dest.exists():
        backup=dest.with_name(dest.name+'.before_corruption_install')
        if not backup.exists():shutil.copy2(dest,backup)
    shutil.copy2(source/name,dest)
dest=target/'util'
if dest.exists():
    backup=target/'util.before_corruption_install'
    if not backup.exists():shutil.copytree(dest,backup)
shutil.copytree(source/'util',dest,dirs_exist_ok=True)
text=init.read_text()
for line in ('from .test_time_aug import MultiScaleFlipAug3D','from .test_time_aug import CorruptionMethods'):
    if line not in text:
        text += '\n' + line + '\n'
init.write_text(text)
print('Installed. Review backups and follow README for dependencies and YAML insertion.')
