import { app } from "scripts/app.js";
import type {
  LGraphCanvas as TLGraphCanvas,
  ContextMenuItem,
  LGraphNode,
} from "typings/litegraph.js";
import type { ComfyNodeConstructor, ComfyObjectInfo } from "typings/comfy.js";
import { rgthree } from "./rgthree.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";

function getOutputNodes(nodes: LGraphNode[]) {
  return (
    nodes?.filter((n) => {
      return (
        n.mode != LiteGraph.NEVER &&
        ((n.constructor as any).nodeData as ComfyObjectInfo)?.output_node
      );
    }) || []
  );
}

function showQueueNodesMenuIfOutputNodesAreSelected(existingOptions: ContextMenuItem[]) {
  if (CONFIG_SERVICE.getConfigValue("features.menu_queue_selected_nodes") === false) {
    return;
  }
  const outputNodes = getOutputNodes(Object.values(app.canvas.selected_nodes));
  const menuItem = {
    content: `Queue Selected Output Nodes (rgthree) &nbsp;`,
    className: "rgthree-contextmenu-item",
    callback: () => {
      rgthree.queueOutputNodes(outputNodes.map((n) => n.id));
    },
    disabled: !outputNodes.length,
  };

  let idx = existingOptions.findIndex((o) => o?.content === "Outputs") + 1;
  idx = idx || existingOptions.findIndex((o) => o?.content === "Align") + 1;
  idx = idx || 3;
  existingOptions.splice(idx, 0, menuItem);
}

function showQueueGroupNodesMenuIfGroupIsSelected(existingOptions: ContextMenuItem[]) {
  if (CONFIG_SERVICE.getConfigValue("features.menu_queue_selected_nodes") === false) {
    return;
  }
  const group =
    rgthree.lastAdjustedMouseEvent &&
    app.graph.getGroupOnPos(
      rgthree.lastAdjustedMouseEvent.canvasX,
      rgthree.lastAdjustedMouseEvent.canvasY,
    );

  const outputNodes = group && getOutputNodes(group._nodes);
  const menuItem = {
    content: `Queue Group Output Nodes (rgthree) &nbsp;`,
    className: "rgthree-contextmenu-item",
    callback: () => {
      outputNodes && rgthree.queueOutputNodes(outputNodes.map((n) => n.id));
    },
    disabled: !outputNodes?.length,
  };

  let idx = existingOptions.findIndex((o) => o?.content?.startsWith("Queue Selected ")) + 1;
  idx = idx || existingOptions.findIndex((o) => o?.content === "Outputs") + 1;
  idx = idx || existingOptions.findIndex((o) => o?.content === "Align") + 1;
  idx = idx || 3;
  existingOptions.splice(idx, 0, menuItem);
}

/**
 * Adds a "Queue Node" menu item to all output nodes, working with `rgthree.queueOutputNode` to
 * execute only a single node's path.
 */
app.registerExtension({
  name: "rgthree.QueueNode",
  async beforeRegisterNodeDef(nodeType: ComfyNodeConstructor, nodeData: ComfyObjectInfo) {
    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (
      canvas: TLGraphCanvas,
      options: ContextMenuItem[],
    ) {
      getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;
      showQueueNodesMenuIfOutputNodesAreSelected(options);
      showQueueGroupNodesMenuIfGroupIsSelected(options);
    };
  },

  async setup() {
    const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function (...args: any[]) {
      const options = getCanvasMenuOptions.apply(this, [...args] as any);
      showQueueNodesMenuIfOutputNodesAreSelected(options);
      showQueueGroupNodesMenuIfGroupIsSelected(options);
      return options;
    };
  },
});
