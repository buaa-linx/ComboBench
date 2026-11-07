# Qualitative videos

Four MP4/WebM positions are reserved, grouped by dataset. Configure the two entries in each array in `site-config.js`:

- `videos.nuscenes[0]`: nuScenes-combo / Fog-Echo Depletion-Occlusion
- `videos.nuscenes[1]`: nuScenes-combo / title to be determined
- `videos.kitti[0]`: KITTI-combo / Fog-Echo Depletion-Occlusion
- `videos.kitti[1]`: KITTI-combo / title to be determined

Keep an entry `null` until its real video is available. Placeholders are not sample videos. Each position can be filled independently. Edit the matching `h4` in `index.html` when a title is finalized.
