import { app } from "../../scripts/app.js";

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
  app.graph?.setDirtyCanvas?.(true, true);
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
    refreshVisibility(node);
    return result;
  };
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
      for (const name of [
        "method",
        "write_thumbs",
        "split_clips",
        "show_all_settings",
      ]) {
        hookWidget(this, name);
      }
      refreshVisibility(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      refreshVisibility(this);
      return result;
    };
  },
});
