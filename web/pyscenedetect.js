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
  if (Array.isArray(message.scene_previews) && message.scene_previews.length) {
    return message.scene_previews;
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

function stylePreviewVideo(video) {
  video.controls = true;
  video.loop = false;
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.disablePictureInPicture = true;
  video.poster =
    "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
  video.style.width = "100%";
  video.style.height = "100%";
  video.style.objectFit = "contain";
  video.style.background = "#111";
  video.style.display = "block";
}

function waitForEventOrTimeout(video, eventName, timeoutMs) {
  return Promise.race([
    new Promise((resolve, reject) => {
      const onError = () => reject(new Error("video failed"));
      const onEvent = () => {
        video.removeEventListener("error", onError);
        resolve();
      };
      video.addEventListener(eventName, onEvent, { once: true });
      video.addEventListener("error", onError, { once: true });
    }),
    new Promise((resolve) => setTimeout(resolve, timeoutMs)),
  ]);
}

async function waitForStartFrame(video) {
  video.pause();
  if (video.readyState < 2) {
    await waitForEventOrTimeout(video, "loadeddata", 4000);
  }
  video.pause();
  try {
    if (video.currentTime !== 0) {
      video.currentTime = 0;
      await waitForEventOrTimeout(video, "seeked", 500);
    }
  } catch {
    return;
  }
}

function playbackLimitSec(video, sceneDuration) {
  const scene = Number(sceneDuration);
  const file = video.duration;
  const limits = [];
  if (Number.isFinite(scene) && scene > 0) {
    limits.push(scene);
  }
  if (Number.isFinite(file) && file > 0) {
    limits.push(file);
  }
  if (!limits.length) {
    return null;
  }
  return Math.min(...limits);
}

function attachLoopFromStart(video, sceneDuration) {
  const restart = (time) => {
    const limit = playbackLimitSec(video, sceneDuration);
    if (limit == null) {
      return;
    }
    if (time >= limit - 1 / 120) {
      video.currentTime = 0;
    }
  };
  const onFrame = (_now, meta) => {
    if (!video.isConnected) {
      return;
    }
    restart(meta?.mediaTime ?? video.currentTime);
    if (typeof video.requestVideoFrameCallback === "function") {
      video.requestVideoFrameCallback(onFrame);
    }
  };
  if (typeof video.requestVideoFrameCallback === "function") {
    video.requestVideoFrameCallback(onFrame);
  } else {
    video.addEventListener("timeupdate", () => restart(video.currentTime));
  }
}

function discardPendingVideos(stage, keep) {
  for (const child of [...stage.children]) {
    if (child === keep) {
      continue;
    }
    unloadVideo(child);
    child.remove();
  }
}

function showPreviewScene(node, index) {
  const entries = node._psdPreviewEntries || [];
  const stage = node._psdPreviewStage;
  if (!entries.length || !stage) {
    return;
  }

  const next = clampSceneIndex(index, entries.length);
  node._psdPreviewIndex = next;
  if (node._psdPreviewIndexInput) {
    node._psdPreviewIndexInput.value = String(next + 1);
  }
  if (node._psdPreviewTotalLabel) {
    node._psdPreviewTotalLabel.textContent = `/ ${entries.length}`;
  }

  const current = node._psdPreviewVideo;
  const currentUrl = current?.getAttribute("src");
  const nextUrl = viewUrl(entries[next]);
  if (current && currentUrl === nextUrl) {
    return;
  }

  if (current) {
    current.pause();
    current.style.visibility = "hidden";
  }

  const token = (node._psdPreviewLoadToken = (node._psdPreviewLoadToken || 0) + 1);
  discardPendingVideos(stage, current);

  const incoming = document.createElement("video");
  stylePreviewVideo(incoming);
  attachLoopFromStart(incoming, entries[next].duration_sec);
  incoming.style.position = "absolute";
  incoming.style.left = "-9999px";
  incoming.style.top = "0";
  incoming.style.width = "100%";
  incoming.style.height = "100%";
  incoming.style.opacity = "0";
  incoming.style.pointerEvents = "none";

  const reveal = () => {
    if (token !== node._psdPreviewLoadToken) {
      unloadVideo(incoming);
      incoming.remove();
      return;
    }
    incoming.style.left = "0";
    incoming.style.right = "0";
    incoming.style.bottom = "0";
    incoming.style.opacity = "1";
    incoming.style.pointerEvents = "";
    const outgoing = node._psdPreviewVideo;
    node._psdPreviewVideo = incoming;
    if (outgoing && outgoing !== incoming) {
      unloadVideo(outgoing);
      outgoing.remove();
    }
    incoming.play().catch(() => {});
  };

  incoming.addEventListener(
    "error",
    () => {
      if (token !== node._psdPreviewLoadToken) {
        incoming.remove();
        return;
      }
      incoming.style.left = "0";
      incoming.style.opacity = "1";
    },
    { once: true }
  );

  stage.appendChild(incoming);
  incoming.src = nextUrl;
  waitForStartFrame(incoming).then(() => {
    if (token !== node._psdPreviewLoadToken) {
      unloadVideo(incoming);
      incoming.remove();
      return;
    }
    reveal();
  });
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
  node._psdPreviewStage = null;
  node._psdPreviewIndexInput = null;
  node._psdPreviewTotalLabel = null;
  node._psdPreviewLoadToken = 0;
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

    const stage = document.createElement("div");
    stage.style.position = "relative";
    stage.style.width = "100%";
    stage.style.height = "220px";
    stage.style.background = "#111";
    stage.style.overflow = "hidden";

    node._psdPreviewStage = stage;
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
    container.append(toolbar, stage);
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

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      hideNativeFirstClipPreview(this);
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted?.apply(this, arguments);
      hideNativeFirstClipPreview(this);
      renderVideoPreviews(this, previewEntries(message));
      return result;
    };
  },
});
