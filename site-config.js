// Temporary public video URLs; run Sync website videos workflow to copy MP4s into this repository.
// Fill public URLs here when the corresponding resources are available.
window.ComboBenchConfig = {
  venue: null,
  links: {paper:"https://github.com/buaa-linx/ComboBench-code",code:"https://github.com/buaa-linx/ComboBench-code",data:"https://huggingface.co/datasets/linchunmian/ComboBench"},
  authorUrls: {chunmian:null,haobing:null,lei:null,xu:null,daxin:null},
  // Video URLs follow the corruption-composition button order for each dataset.
  videos: {nuscenes:["assets/videos/fog_occlusion_echo.mp4","assets/videos/snow_beam_cutout.mp4"],kitti:["assets/videos/fog_occlusion_echo_kitti.mp4","assets/videos/snow_beam_cutout_kitti.mp4"]}
};
