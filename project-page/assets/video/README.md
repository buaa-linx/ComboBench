# Qualitative videos

Qualitative Results uses one shared player. Select a dataset, then a corruption composition.

| Dataset | Composition | File in `assets/videos/` |
| --- | --- | --- |
| nuScenes-combo | Fog-Echo Depletion-Occlusion | `fog_occlusion_echo.mp4` |
| nuScenes-combo | Snow-Beam Omission-Cluster Cutout | `snow_beam_cutout.mp4` |
| KITTI-combo | Fog-Echo Depletion-Occlusion | `fog_occlusion_echo_kitti.mp4` |
| KITTI-combo | Snow-Beam Omission-Cluster Cutout | `snow_beam_cutout_kitti.mp4` |

Configure the corresponding URLs in `videos` in `site-config.js`. Only the selected video is loaded; changing the selection pauses and releases the previous source. The player uses the video directly without separate poster images. Titles are defined by the composition buttons in `index.html`.
