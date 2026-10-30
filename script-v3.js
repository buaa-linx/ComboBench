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
      document.getElementById("resource-dialog-message").textContent = `A public ${resourceNames[key].toLowerCase()} URL has not been added to this draft. This entry is reserved for the corresponding resource.`;
      resourceDialog.showModal();document.getElementById("resource-dismiss").focus();
    });
  }
});
document.getElementById("resource-dialog-close").addEventListener("click",()=>resourceDialog.close());
document.getElementById("resource-dismiss").addEventListener("click",()=>resourceDialog.close());
resourceDialog.addEventListener("close",()=>resourceTrigger?.focus());

if (typeof config.venue === "string" && config.venue.trim()) document.getElementById("venue-line").textContent = config.venue.trim();
document.querySelectorAll("[data-author]").forEach(element => {
  const url = safeUrl(config.authorUrls?.[element.dataset.author]);
  if (!url) return;
  const link = document.createElement("a");link.href = url;link.target = "_blank";link.rel = "noopener";link.textContent = element.textContent;element.replaceChildren(link);
});
document.querySelectorAll("[data-video]").forEach(slot => {
  const key = slot.dataset.video;
  const url = safeUrl(config.videos?.[key]);
  if (!url) return;
  const video = document.createElement("video");
  video.controls = true;video.playsInline = true;video.preload = "metadata";
  video.src = url;video.setAttribute("aria-label",`${key === "nuscenes" ? "nuScenes" : "KITTI"} demonstration video`);
  slot.replaceChildren(video);
  document.getElementById(`video-state-${key}`).textContent = key === "nuscenes" ? "nuScenes-Combo demonstration" : "KITTI-Combo demonstration";
});

const imageDialog = document.getElementById("image-dialog");
let imageTrigger;
document.querySelectorAll(".figure-zoom").forEach(button => button.addEventListener("click",()=> {
  if (typeof imageDialog.showModal !== "function") {window.open(button.dataset.image,"_blank","noopener");return;}
  imageTrigger = button;
  const image = document.getElementById("dialog-image");image.src = button.dataset.image;image.alt = button.querySelector("img").alt;
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
