"use strict";
const config = window.ComboBenchConfig || {};
const safeUrl = value => typeof value === "string" && value.trim() && !/^(?:javascript|data|vbscript):/i.test(value.trim()) ? value.trim() : null;
const resourceNames = {paper:"Paper",code:"Code",data:"Data"};
const resourceDialog = document.getElementById("resource-dialog");
let resourceTrigger;
document.querySelectorAll("[data-resource]").forEach(element => {
  const key = element.dataset.resource;
  const url = safeUrl(config.links?.[key]);
  if (url) {
    let link = element;
    if (element.tagName !== "A") {
      link = document.createElement("a");
      link.className = element.className;
      link.innerHTML = element.innerHTML;
      link.dataset.resource = key;
      element.replaceWith(link);
    }
    link.href = url; link.target = "_blank"; link.rel = "noopener";
    link.classList.remove("is-pending"); link.querySelector("small")?.remove();
  } else {
    element.setAttribute("aria-label",`${resourceNames[key]} — link pending`);
    element.addEventListener("click",event => {
      event.preventDefault();resourceTrigger = element;
      document.getElementById("resource-dialog-title").textContent = `${resourceNames[key]} link pending`;
      document.getElementById("resource-dialog-message").textContent = `The public ${resourceNames[key].toLowerCase()} link will be added here when the resource is available.`;
      resourceDialog.showModal();document.getElementById("resource-dismiss").focus();
    });
  }
});
document.getElementById("resource-dialog-close").addEventListener("click",()=>resourceDialog.close());
document.getElementById("resource-dismiss").addEventListener("click",()=>resourceDialog.close());
resourceDialog.addEventListener("close",()=>resourceTrigger?.focus());

const venueLine = document.getElementById("venue-line");
if (venueLine && typeof config.venue === "string" && config.venue.trim()) {
  venueLine.textContent = config.venue.trim();
  venueLine.hidden = false;
}
document.querySelectorAll("[data-author]").forEach(element => {
  const url = safeUrl(config.authorUrls?.[element.dataset.author]);
  if (!url) return;
  const link = document.createElement("a");link.href = url;link.target = "_blank";link.rel = "noopener";link.textContent = element.textContent;element.replaceChildren(link);
});
// One shared player; only the selected composition is loaded.
const videoExplorer = document.querySelector("[data-video-explorer]");
if (videoExplorer) {
  const datasetButtons = Array.from(videoExplorer.querySelectorAll("[data-dataset]"));
  const compositionButtons = Array.from(videoExplorer.querySelectorAll("[data-corruption-index]"));
  const player = document.getElementById("qualitative-player");
  const empty = document.getElementById("qualitative-empty");
  const panel = document.getElementById("qualitative-player-panel");
  const activeDataset = document.getElementById("active-dataset");
  const activeTitle = document.getElementById("active-video-title");
  const fallbackUrl = player.getAttribute("src");
  const fallbackPoster = player.getAttribute("poster");
  let selectedDataset = "nuscenes";
  let selectedComposition = 0;

  const renderVideo = () => {
    const datasetButton = datasetButtons.find(button => button.dataset.dataset === selectedDataset);
    const compositionButton = compositionButtons[selectedComposition];
    const datasetName = datasetButton.textContent.trim();
    const title = compositionButton.textContent.trim();
    const urls = config.videos?.[selectedDataset];
    const configuredUrl = Array.isArray(urls) ? urls[selectedComposition] : null;
    const url = safeUrl(configuredUrl) || (!config.videos && selectedDataset === "nuscenes" && selectedComposition === 0 ? fallbackUrl : null);
    const poster = safeUrl(config.videoPosters?.[selectedDataset]?.[selectedComposition]) || (!config.videoPosters && selectedDataset === "nuscenes" && selectedComposition === 0 ? fallbackPoster : null);

    datasetButtons.forEach(button => {
      const selected = button === datasetButton;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    compositionButtons.forEach(button => button.setAttribute("aria-pressed", String(button === compositionButton)));
    panel.setAttribute("aria-labelledby", datasetButton.id);
    activeDataset.textContent = datasetName;
    activeTitle.textContent = title;
    player.setAttribute("aria-label", `${datasetName}: ${title}`);

    if (player.getAttribute("src") !== url) {
      player.pause();
      player.removeAttribute("src");
      player.removeAttribute("poster");
      if (url) player.setAttribute("src", url);
      if (url && poster) player.setAttribute("poster", poster);
      player.load();
    } else if (url && poster) {
      player.setAttribute("poster", poster);
    }
    player.hidden = !url;
    empty.hidden = Boolean(url);
  };

  const chooseDataset = dataset => {
    if (selectedDataset === dataset) return;
    selectedDataset = dataset;
    renderVideo();
  };
  datasetButtons.forEach((button, index) => {
    button.addEventListener("click", () => chooseDataset(button.dataset.dataset));
    button.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % datasetButtons.length;
      if (event.key === "ArrowLeft") next = (index - 1 + datasetButtons.length) % datasetButtons.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = datasetButtons.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      chooseDataset(datasetButtons[next].dataset.dataset);
      datasetButtons[next].focus();
    });
  });
  compositionButtons.forEach(button => button.addEventListener("click", () => {
    const next = Number(button.dataset.corruptionIndex);
    if (selectedComposition === next) return;
    selectedComposition = next;
    renderVideo();
  }));
  renderVideo();
  videoExplorer.classList.add("is-enhanced");
  videoExplorer.querySelector(".dataset-tabs").hidden = false;
  videoExplorer.querySelector(".corruption-tabs").hidden = false;
}

const imageDialog = document.getElementById("image-dialog");
let imageTrigger;
document.querySelectorAll(".figure-zoom").forEach(button => button.addEventListener("click",()=> {
  if (typeof imageDialog.showModal !== "function") {window.open(button.dataset.image,"_blank","noopener");return;}
  imageTrigger = button;
  const image = document.getElementById("dialog-image");image.src = button.dataset.image;image.alt = button.querySelector("img").alt;image.classList.toggle("is-table",button.classList.contains("paper-table-zoom"));
  document.getElementById("dialog-caption").textContent = button.dataset.caption;
  imageDialog.showModal();document.getElementById("dialog-close").focus();
}));
document.getElementById("dialog-close").addEventListener("click",()=>imageDialog.close());
imageDialog.addEventListener("close",()=>imageTrigger?.focus());
[imageDialog,resourceDialog].forEach(dialog => dialog.addEventListener("click",event => {
  if (event.target !== dialog) return;
  const box = dialog.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
}));

const copyButton = document.getElementById("copy-citation");
let copyTimer;
copyButton.addEventListener("click",async ()=> {
  const citation = document.getElementById("bibtex").textContent.trim();
  let copied = false;
  try {if (navigator.clipboard?.writeText) {await navigator.clipboard.writeText(citation);copied = true;}} catch {}
  if (!copied) {
    const field = document.createElement("textarea");field.value = citation;field.setAttribute("readonly","");field.style.cssText = "position:fixed;left:-9999px;top:0";document.body.appendChild(field);field.select();
    try {copied = document.execCommand("copy");} catch {}
    field.remove();copyButton.focus();
  }
  clearTimeout(copyTimer);copyButton.textContent = copied ? "Copied ✓" : "Select text to copy";
  document.getElementById("copy-status").textContent = copied ? "BibTeX copied to clipboard." : "Select the BibTeX text and copy it manually.";
  copyTimer = setTimeout(()=>{copyButton.textContent = "Copy BibTeX";},3000);
});
