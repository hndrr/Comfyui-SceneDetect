import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/detector_settings.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/, "");

for (const name of ["PySceneDetectVideo", "PySceneDetectToImages"]) {
    test(`${name} preserves widgets and callbacks while toggling details`, () => {
        let extension;
        vm.runInNewContext(source, {
            app: { registerExtension(value) { extension = value; } },
        });
        const originalSize = () => [200, 20];
        let calls = 0;
        class Node {
            size = [300, 96];
            widgets = [
                { name: "method", value: "content", options: {} },
                { name: "hash_threshold", value: 0.395, options: {} },
                { name: "downscale", value: 2, options: {} },
                { name: "detector_settings", value: "default", options: {},
                  callback() { calls++; return "callback"; } },
                { name: "adaptive_threshold", value: 3, options: {}, computeSize: originalSize },
                { name: "fade_bias", value: 0, options: {} },
            ];
            onNodeCreated() { return "created"; }
            onConfigure() { return "configured"; }
            onWidgetChanged() { calls++; return "changed"; }
            computeSize() { return [300, this.widgets.filter(w => !w.hidden).length * 24]; }
            setSize(size) { this.size = size; }
            setDirtyCanvas() {}
        }
        extension.beforeRegisterNodeDef(Node, { name });
        const node = new Node();
        const widgets = [...node.widgets];
        assert.equal(node.onNodeCreated(), "created");
        assert.equal(node.size[1], 96);
        assert.equal(node.widgets[4].hidden, true);
        assert.equal(node.widgets[4].options.hidden, true);

        const control = node.widgets[3];
        for (let i = 0; i < 3; i++) {
            control.value = "custom";
            assert.equal(control.callback(), "callback");
            assert.equal(node.size[1], 144);
            assert.equal(node.widgets[4].computeSize, originalSize);
            assert.equal(node.widgets[4].options.hidden, false);
            assert.equal(node.widgets[1].options.hidden, undefined);
            assert.equal(node.widgets[2].options.hidden, undefined);
            control.value = "default";
            assert.equal(control.callback(), "callback");
            assert.equal(node.size[1], 96);
        }
        assert.equal(calls, 6);
        assert.deepEqual(node.widgets, widgets);

        control.value = "custom";
        node.size = [500, 700];
        assert.equal(node.onConfigure(), "configured");
        assert.deepEqual(Array.from(node.size), [500, 700]);
        assert.equal(node.widgets[4].hidden, false);
        control.value = "default";
        assert.equal(node.onConfigure(), "configured");
        assert.deepEqual(Array.from(node.size), [500, 700]);
        assert.equal(node.widgets[4].hidden, true);
        control.value = "custom";
        assert.equal(control.callback(), "callback");
        assert.equal(calls, 7);
        assert.deepEqual(Array.from(node.size), [500, 144]);

        node.size = [500, 96];
        assert.equal(node.onConfigure(), "configured");
        assert.deepEqual(Array.from(node.size), [500, 144]);

        control.callback = () => "renderer callback";
        control.value = "default";
        assert.equal(node.onWidgetChanged("detector_settings"), "changed");
        assert.equal(calls, 8);
        assert.equal(node.size[1], 96);
        assert.equal(node.size[0], 500);
        assert.equal(node.widgets.length, 6);
        assert.equal(node.widgets[4].value, 3);
    });
}
