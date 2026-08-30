import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASSES = new Set(["PySceneDetectVideo", "PySceneDetectToImages"]);

const METHOD_WIDGETS = {
  content: [
    "threshold",
    "luma_only",
    "delta_hue",
    "delta_sat",
    "delta_lum",
    "delta_edges",
    "kernel_size",
  ],
  adaptive: [
    "adaptive_threshold",
    "window_width",
    "min_content_val",
    "luma_only",
    "delta_hue",
    "delta_sat",
    "delta_lum",
    "delta_edges",
    "kernel_size",
  ],
  threshold: ["threshold", "fade_bias", "add_final_scene", "threshold_method"],
  hash: ["hash_threshold", "hash_size", "hash_lowpass"],
  histogram: ["hist_threshold", "hist_bins"],
};

const METHOD_WIDGET_NAMES = new Set(Object.values(METHOD_WIDGETS).flat());

const ALWAYS_VISIBLE = new Set([
  "method",
  "min_scene_len_sec",
  "min_scene_len_frames",
  "representative",
  "max_width",
  "max_height",
  "limit_scenes",
  "write_thumbs",
  "split_clips",
  "prompt_template",
  "start_in_scene",
  "downscale",
  "show_all_settings",
]);

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function widgetValue(node, name) {
  return widgetByName(node, name)?.value;
}

function isTruthy(value) {
  return value === true || value === 1 || value === "true";
}

function setWidgetHidden(widget, hidden) {
  if (!widget) {
    return;
  }
  if (widget._psdOrigComputeSize === undefined) {
    widget._psdOrigComputeSize = widget.computeSize;
  }
  widget.hidden = hidden;
  widget.computeSize = hidden
    ? () => [0, -4]
    : widget._psdOrigComputeSize;

  const nextOptions = { ...(widget.options || {}), hidden };
  widget.options = nextOptions;
  if (widget._state && typeof widget._state === "object") {
    widget._state.options = { ...(widget._state.options || {}), hidden };
  }
  updateStoreOptions(widget, { hidden });

  for (const key of ["element", "inputEl"]) {
    const el = widget[key];
    if (el?.style) {
      el.style.display = hidden ? "none" : "";
    }
  }
}

function getPinia() {
  const vueApp =
    app.vueApp ||
    document.querySelector("#vue-app")?.__vue_app__ ||
    document.querySelector("#app")?.__vue_app__;
  return (
    vueApp?.config?.globalProperties?.$pinia ||
    vueApp?._context?.provides?.pinia ||
    null
  );
}

function getWidgetValueStore() {
  const pinia = getPinia();
  if (!pinia) {
    return null;
  }
  if (typeof pinia._s?.get === "function" && pinia._s.has("widgetValue")) {
    return pinia._s.get("widgetValue");
  }
  if (pinia._s) {
    for (const store of pinia._s.values()) {
      if (
        typeof store?.updateOptions === "function" &&
        typeof store?.getWidget === "function"
      ) {
        return store;
      }
    }
  }
  return null;
}

function updateStoreOptions(widget, patch) {
  const widgetId = widget.widgetId;
  if (typeof widgetId !== "string" || !widgetId) {
    return;
  }
  const store = getWidgetValueStore();
  if (!store) {
    return;
  }
  if (typeof store.updateOptions === "function") {
    store.updateOptions(widgetId, patch);
    return;
  }
  const state = store.getWidget?.(widgetId);
  if (state) {
    state.options = { ...(state.options || {}), ...patch };
  }
}

function refreshVisibility(node) {
  const showAll = isTruthy(widgetValue(node, "show_all_settings"));
  const method = widgetValue(node, "method") || "content";
  const visibleForMethod = new Set(METHOD_WIDGETS[method] || []);
  const writeThumbs = isTruthy(widgetValue(node, "write_thumbs"));
  const splitClips = isTruthy(widgetValue(node, "split_clips"));

  for (const widget of node.widgets || []) {
    if (ALWAYS_VISIBLE.has(widget.name)) {
      setWidgetHidden(widget, false);
      continue;
    }
    if (widget.name === "thumbs_dir") {
      setWidgetHidden(widget, !(showAll || writeThumbs));
      continue;
    }
    if (widget.name === "split_reencode") {
      setWidgetHidden(widget, !(showAll || splitClips));
      continue;
    }
    if (METHOD_WIDGET_NAMES.has(widget.name)) {
      setWidgetHidden(widget, !(showAll || visibleForMethod.has(widget.name)));
      continue;
    }
  }

  const size = node.computeSize();
  node.setSize([node.size[0], size[1]]);
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
}

function visibilityKey(node) {
  return [
    node.widgets?.length || 0,
    widgetValue(node, "method") || "content",
    isTruthy(widgetValue(node, "show_all_settings")) ? "1" : "0",
    isTruthy(widgetValue(node, "write_thumbs")) ? "1" : "0",
    isTruthy(widgetValue(node, "split_clips")) ? "1" : "0",
  ].join("|");
}

function syncVisibility(node) {
  const key = visibilityKey(node);
  if (node._psdVisibilityKey === key) {
    return;
  }
  node._psdVisibilityKey = key;
  refreshVisibility(node);
}

function hookWidget(node, name) {
  const widget = widgetByName(node, name);
  if (!widget || widget._psdVisibilityHooked) {
    return;
  }
  widget._psdVisibilityHooked = true;
  const original = widget.callback;
  widget.callback = function () {
    const result = original?.apply(this, arguments);
    node._psdVisibilityKey = undefined;
    syncVisibility(node);
    return result;
  };

  const proto = Object.getPrototypeOf(widget);
  const descriptor =
    Object.getOwnPropertyDescriptor(widget, "value") ||
    (proto && Object.getOwnPropertyDescriptor(proto, "value"));
  if (descriptor?.get && descriptor?.set && !widget._psdValueHooked) {
    widget._psdValueHooked = true;
    Object.defineProperty(widget, "value", {
      configurable: true,
      enumerable: true,
      get() {
        return descriptor.get.call(this);
      },
      set(value) {
        descriptor.set.call(this, value);
        node._psdVisibilityKey = undefined;
        syncVisibility(node);
      },
    });
  }
}

function hookVisibilityWidgets(node) {
  for (const name of [
    "method",
    "write_thumbs",
    "split_clips",
    "show_all_settings",
  ]) {
    hookWidget(node, name);
  }
}

app.registerExtension({
  name: "Comfyui-SceneDetect.WidgetVisibility",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_CLASSES.has(nodeData.name)) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      hookVisibilityWidgets(this);
      this._psdVisibilityKey = undefined;
      syncVisibility(this);
      return result;
    };

    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      const result = onAdded?.apply(this, arguments);
      hookVisibilityWidgets(this);
      this._psdVisibilityKey = undefined;
      syncVisibility(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      hookVisibilityWidgets(this);
      this._psdVisibilityKey = undefined;
      syncVisibility(this);
      return result;
    };

    const onWidgetChanged = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function () {
      const result = onWidgetChanged?.apply(this, arguments);
      this._psdVisibilityKey = undefined;
      syncVisibility(this);
      return result;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
      const result = onDrawForeground?.apply(this, arguments);
      hookVisibilityWidgets(this);
      syncVisibility(this);
      return result;
    };
  },
  nodeCreated(node) {
    if (!NODE_CLASSES.has(node.comfyClass)) {
      return;
    }
    hookVisibilityWidgets(node);
    node._psdVisibilityKey = undefined;
    syncVisibility(node);
  },
});

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
      "No videos to preview. Enable split_clips and connect videos.";
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
