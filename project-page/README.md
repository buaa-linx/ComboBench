# ComboBench

ComboBench studies robust 3D perception under compositional camera–LiDAR corruptions. The project page introduces the configurable ComboGen pipeline, the CARE framework, corrupted observation examples, and experimental results on nuScenes-Combo and KITTI-Combo.

## Project page

[Visit the project page](https://buaa-linx.github.io/ComboBench/)

## Resource status

- Paper PDF: not publicly released yet.
- Code button: [ComboBench on Hugging Face](https://huggingface.co/datasets/linchunmian/ComboBench).
- The Paper button currently links to the official implementation repository.
- The data button is reserved until a public URL is available.
- Qualitative Results: two nuScenes-combo videos in a side-by-side layout; two KITTI-combo video positions remain reserved.

## Website source

The page is a static website. Its entry point is `index.html`; all assets use relative paths compatible with a GitHub Pages project site.

Resource links, author URLs, publication metadata, and independent video file URLs are configured in `site-config.js`. Keep unavailable URLs as `null`.

The page presents corruption samples, the CARE architecture, experimental results with the original Figure 5 and Tables 7 and 8, and qualitative detection results.

The complete numerical data from Tables 7 and 8 is also retained in `assets/data/table7-table8.json`. Figure and data provenance is recorded in `SOURCE_NOTES.md`.

## Third-party assets

Icon sources and license notices are preserved in [assets/icons/SOURCES.md](assets/icons/SOURCES.md).

Typography uses locally hosted Noto Sans under the SIL Open Font License; source and license are in `assets/fonts/`.

## Acknowledgements

The typography and video presentation of this project page were inspired by [Academic Project Astro Template](https://github.com/RomanHauksson/academic-project-astro-template) by Roman Hauksson.

## Deployment

The project page is published from this `project-page/` directory by `.github/workflows/pages.yml` in the ComboBench repository. Set Settings → Pages → Source to GitHub Actions.
