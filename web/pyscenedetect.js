import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

function viewUrl(entry) {
  const params = new URLSearchParams({
    filename: entry.filename,
    type: entry.type || "temp",
    subfolder: entry.subfolder || "",
  });
  return api.apiURL(`/view?${params.toString()}`);
}

function previewEntries(message) {
  if (!message) {
    return [];
  }
  if (Array.isArray(message.videos) && message.videos.length) {
    return message.videos;
  }
  if (Array.isArray(message.images)) {
    return message.images;
  }
  return [];
}

function ensureVideoPreviewWidget(node) {
  if (node._psdVideoPreviewContainer) {
    return node._psdVideoPreviewContainer;
  }

  const container = document.createElement("div");
  container.className = "psd-video-preview";
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.gap = "8px";
  container.style.width = "100%";

  const widget = node.addDOMWidget("psd-video-preview", "div", container, {
    serialize: false,
    hideOnZoom: false,
  });
  widget.computeSize = function (width) {
    return [width, node._psdVideoPreviewHeight || 28];
  };

  node._psdVideoPreviewContainer = container;
  node._psdVideoPreviewWidget = widget;
  return container;
}

function hideNativeFirstClipPreview(node) {
  for (const widget of node.widgets || []) {
    if (widget.name === "video-preview") {
      widget.hidden = true;
      widget.computeSize = () => [0, -4];
    }
  }
}

function unloadVideo(video) {
  if (!video) {
    return;
  }
  video.pause();
  video.removeAttribute("src");
  video.load();
}

function clampSceneIndex(index, count) {
  if (count <= 0) {
    return 0;
  }
  return ((index % count) + count) % count;
}

function showPreviewScene(node, index) {
  const entries = node._psdPreviewEntries || [];
  if (!entries.length || !node._psdPreviewVideo) {
    return;
  }

  const next = clampSceneIndex(index, entries.length);
  node._psdPreviewIndex = next;
  unloadVideo(node._psdPreviewVideo);
  node._psdPreviewVideo.src = viewUrl(entries[next]);
  if (node._psdPreviewIndexInput) {
    node._psdPreviewIndexInput.value = String(next + 1);
  }
  if (node._psdPreviewTotalLabel) {
    node._psdPreviewTotalLabel.textContent = `/ ${entries.length}`;
  }
}

function stopWidgetEvent(event) {
  event.stopPropagation();
}

function renderVideoPreviews(node, entries) {
  const container = ensureVideoPreviewWidget(node);
  unloadVideo(node._psdPreviewVideo);
  container.replaceChildren();
  node._psdPreviewEntries = entries;
  node._psdPreviewIndex = 0;
  node._psdPreviewVideo = null;
  node._psdPreviewIndexInput = null;
  node._psdPreviewTotalLabel = null;
  hideNativeFirstClipPreview(node);

  if (!entries.length) {
    node._psdVideoPreviewHeight = 28;
    const empty = document.createElement("div");
    empty.textContent =
      "No videos to preview. Enable split_clips and connect scene_videos.";
    empty.style.opacity = "0.65";
    empty.style.padding = "6px 4px";
    empty.style.fontSize = "12px";
    container.appendChild(empty);
  } else {
    node._psdVideoPreviewHeight = 280;

    const toolbar = document.createElement("div");
    toolbar.style.display = "flex";
    toolbar.style.alignItems = "center";
    toolbar.style.gap = "6px";
    toolbar.style.fontSize = "12px";
    toolbar.addEventListener("pointerdown", stopWidgetEvent);

    const prev = document.createElement("button");
    prev.type = "button";
    prev.textContent = "◀";
    prev.title = "Previous scene";

    const next = document.createElement("button");
    next.type = "button";
    next.textContent = "▶";
    next.title = "Next scene";

    const input = document.createElement("input");
    input.type = "number";
    input.min = "1";
    input.max = String(entries.length);
    input.step = "1";
    input.value = "1";
    input.style.width = "4.5em";
    input.title = "Scene number";

    const total = document.createElement("span");
    total.textContent = `/ ${entries.length}`;
    total.style.opacity = "0.8";

    const video = document.createElement("video");
    video.controls = true;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.style.width = "100%";
    video.style.maxHeight = "220px";
    video.style.background = "#111";

    node._psdPreviewVideo = video;
    node._psdPreviewIndexInput = input;
    node._psdPreviewTotalLabel = total;

    prev.addEventListener("click", (event) => {
      stopWidgetEvent(event);
      showPreviewScene(node, (node._psdPreviewIndex || 0) - 1);
    });
    next.addEventListener("click", (event) => {
      stopWidgetEvent(event);
      showPreviewScene(node, (node._psdPreviewIndex || 0) + 1);
    });
    input.addEventListener("change", (event) => {
      stopWidgetEvent(event);
      const value = Number(input.value);
      if (Number.isFinite(value)) {
        showPreviewScene(node, Math.round(value) - 1);
      }
    });

    toolbar.append(prev, input, total, next);
    container.append(toolbar, video);
    showPreviewScene(node, 0);
  }

  const size = node.computeSize?.() || node.size;
  if (size) {
    node.setSize([node.size[0], Math.max(node.size[1], size[1])]);
  }
  app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "Comfyui-SceneDetect.PreviewVideos",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "PySceneDetectPreviewVideos") {
      return;
    }

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted?.apply(this, arguments);
      renderVideoPreviews(this, previewEntries(message));
      return result;
    };
  },
});
