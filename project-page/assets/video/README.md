# Qualitative videos

Qualitative Results uses one shared player. First select nuScenes-combo or KITTI-combo, then select a corruption composition:

- Fog-Echo Depletion-Occlusion
- Snow-Beam Omission-Cluster Cutout

The two nuScenes-combo files and their JPEG posters are in `assets/videos/`. Both KITTI-combo entries remain empty until videos are supplied.

Configure the corresponding entries in `videos` and `videoPosters` in `site-config.js`; use `null` for unavailable media. Only the selected video is loaded, and changing selections pauses and releases the previous source. Video titles are defined by the composition buttons in `index.html`.
