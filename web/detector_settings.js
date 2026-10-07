import { app } from "../../scripts/app.js";

const detailNames = new Set([
    "adaptive_threshold", "window_width", "min_content_val",
    "delta_hue", "delta_sat", "delta_lum", "delta_edges", "kernel_size",
    "fade_bias", "add_final_scene", "threshold_method",
]);
const originalSizes = new WeakMap();

function updateDetails(node, preserveSize = false) {
    const custom = node.widgets?.find(w => w.name === "detector_settings")?.value === "custom";
    for (const widget of node.widgets ?? []) {
        if (!detailNames.has(widget.name)) continue;
        if (!originalSizes.has(widget)) originalSizes.set(widget, widget.computeSize);
        widget.hidden = !custom;
        widget.options ??= {};
        widget.options.hidden = !custom;
        if (custom) {
            const size = originalSizes.get(widget);
            if (size) widget.computeSize = size;
            else delete widget.computeSize;
        } else {
            widget.computeSize = () => [0, -4];
        }
    }
    const size = node.computeSize();
    size[0] = Math.max(node.size[0], size[0]);
    if (preserveSize) size[1] = Math.max(node.size[1], size[1]);
    node.setSize(size);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "SceneDetect.DetectorSettings",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["PySceneDetectVideo", "PySceneDetectToImages"].includes(nodeData.name)) return;
        for (const hook of ["onNodeCreated", "onConfigure"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                const control = this.widgets?.find(w => w.name === "detector_settings");
                if (control && !control._sceneDetectWrapped) {
                    const controlNode = this;
                    const callback = control.callback;
                    control.callback = function (...values) {
                        const result = callback?.apply(this, values);
                        updateDetails(controlNode);
                        return result;
                    };
                    control._sceneDetectWrapped = true;
                }
                updateDetails(this, hook === "onConfigure");
                return result;
            };
        }
        const original = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, ...args) {
            const result = original?.call(this, name, ...args);
            if (name === "detector_settings") updateDetails(this);
            return result;
        };
    },
});
