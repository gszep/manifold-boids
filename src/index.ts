import GUI from "lil-gui";
import {
  addEventListeners,
  configureCanvas,
  createPipelineLayout,
  createRenderPipeline,
  createShader,
  getRandomValues,
  renderPass,
  requestDevice,
  setupTextures,
} from "./utils";
import { Struct, bindingsFromWGSL } from "./wgsl";

import computeShader from "./shaders/compute.wgsl";
import renderShader from "./shaders/render.wgsl";

import bindings from "./shaders/includes/bindings.wgsl";
import canvas from "./shaders/includes/canvas.wgsl";
import controls from "./shaders/includes/controls.wgsl";
import interactions from "./shaders/includes/interactions.wgsl";
import dataPoints from "./shaders/includes/data_points.wgsl";
import random from "./shaders/includes/random.wgsl";
import textures from "./shaders/includes/textures.wgsl";

const shaderIncludes: Record<string, string> = {
  data_points: dataPoints,
  random: random,
  canvas: canvas,
  controls: controls,
  bindings: bindings,
  textures: textures,
  interactions: interactions,
};

const DATA_POINT_COUNT = 10000;
const WORKGROUP_SIZE = 256;
const FEATURE_DIMENSION = 3;
const GMM_COMPONENTS = 3; // Number of Gaussian mixture components

// Inject constants into shader includes
shaderIncludes.data_points = shaderIncludes.data_points.replaceAll(
  "{{FEATURE_DIMENSION}}",
  FEATURE_DIMENSION.toString()
);

async function main() {
  const device = await requestDevice({}, [], {
    maxStorageBufferBindingSize: 4294967292,
    maxBufferSize: 4294967292,
  });
  const { context, format, size } = configureCanvas(device);

  // binding indexes matching `shaders/includes/bindings.wgsl`
  const GROUP_INDEX = 0;
  const BINDINGS = bindingsFromWGSL(shaderIncludes.bindings);

  const textures = setupTextures(
    device,
    /*bindings=*/ Object.values(BINDINGS[GROUP_INDEX].TEXTURE),
    /*data=*/ {},
    /*size=*/ {
      width: size.width,
      height: size.height,
    },
    /*format=*/ {
      [BINDINGS[GROUP_INDEX].TEXTURE.INDEX]: "r32uint",
      [BINDINGS[GROUP_INDEX].TEXTURE.RECENCY]: "r32float",
      [BINDINGS[GROUP_INDEX].TEXTURE.DENSITY]: "r32float",
    }
  );

  const canvas = new Struct(shaderIncludes.canvas, device, {
    label: "Canvas",
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const interactions = new Struct(shaderIncludes.interactions, device, {
    label: "Interactions",
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const controls = new Struct(shaderIncludes.controls, device, {
    label: "Controls",
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const dataPoints = new Struct(shaderIncludes.data_points, device, {
    label: "DataPoints",
    size: DATA_POINT_COUNT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const random = new Struct(shaderIncludes.random, device, {
    label: "Random",
    size: DATA_POINT_COUNT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  random.x = getRandomValues(DATA_POINT_COUNT);
  random.y = getRandomValues(DATA_POINT_COUNT);
  random.z = getRandomValues(DATA_POINT_COUNT);
  random.w = getRandomValues(DATA_POINT_COUNT);

  const buffers = {
    [BINDINGS[GROUP_INDEX].BUFFER.CANVAS]: {
      buffer: canvas._gpubuffer,
      type: "uniform" as GPUBufferBindingType,
    },
    [BINDINGS[GROUP_INDEX].BUFFER.INTERACTIONS]: {
      buffer: interactions._gpubuffer,
      type: "uniform" as GPUBufferBindingType,
    },
    [BINDINGS[GROUP_INDEX].BUFFER.CONTROLS]: {
      buffer: controls._gpubuffer,
      type: "uniform" as GPUBufferBindingType,
    },
    [BINDINGS[GROUP_INDEX].BUFFER.DATA_POINTS]: {
      buffer: dataPoints._gpubuffer,
      type: "storage" as GPUBufferBindingType,
    },
    [BINDINGS[GROUP_INDEX].BUFFER.RANDOM]: {
      buffer: random._gpubuffer,
      type: "storage" as GPUBufferBindingType,
    },
  };

  canvas.key = [0, 0];
  canvas.size = [size.width, size.height];
  addEventListeners(interactions, context.canvas, textures.size);

  // overall memory layout
  const pipeline = createPipelineLayout(device, BINDINGS[GROUP_INDEX], textures, buffers);

  const processedComputeShader = computeShader.replaceAll(
    "{{FEATURE_DIMENSION}}",
    FEATURE_DIMENSION.toString()
  );
  const processedRenderShader = renderShader;

  // traditional render pipeline of vert -> frag
  const render = await createRenderPipeline(
    device,
    format,
    pipeline.layout,
    processedRenderShader,
    shaderIncludes
  );

  // Create compute pipelines
  const module = await createShader(device, processedComputeShader, shaderIncludes);

  const update_positions = device.createComputePipeline({
    layout: pipeline.layout,
    compute: { module: module, entryPoint: "update_positions" },
  });

  const update_textures = device.createComputePipeline({
    layout: pipeline.layout,
    compute: { module: module, entryPoint: "update_textures" },
  });

  const WORKGROUP_COUNT_BUFFER = Math.ceil(DATA_POINT_COUNT / WORKGROUP_SIZE);
  const WORKGROUP_COUNT_TEXTURE: [number, number] = [
    Math.ceil(textures.size.width / Math.sqrt(WORKGROUP_SIZE)),
    Math.ceil(textures.size.height / Math.sqrt(WORKGROUP_SIZE)),
  ];

  /**
   * Gaussian mixture model component
   */
  interface GaussianComponent {
    mean: number[];
    stdDev: number;
  }

  /**
   * Initialize Gaussian mixture model with random means and standard deviations
   */
  const initializeGMM = (numComponents: number, dimension: number): GaussianComponent[] => {
    const components: GaussianComponent[] = [];
    for (let i = 0; i < numComponents; i++) {
      const mean = Array.from({ length: dimension }, () => Math.random());
      const stdDev = Math.random() * 0.5 + 0.1; // Range [0.1, 0.6]
      components.push({ mean, stdDev });
    }
    return components;
  };

  /**
   * Box-Muller transform: generate standard normal sample
   */
  const sampleGaussian = (): number => {
    const u1 = Math.random();
    const u2 = Math.random();
    // Clamp u1 to avoid log(0)
    const r = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10)));
    const theta = 2 * Math.PI * u2;
    return r * Math.cos(theta);
  };

  /**
   * Sample from a single Gaussian component
   */
  const sampleFromComponent = (component: GaussianComponent): number[] => {
    return component.mean.map((mu) => mu + component.stdDev * sampleGaussian());
  };

  /**
   * Sample from the mixture: select component uniformly, then sample from it
   * Returns both the component index and the sampled features
   */
  const sampleFromGMM = (
    components: GaussianComponent[]
  ): { componentIdx: number; features: number[] } => {
    const componentIdx = Math.floor(Math.random() * components.length);
    const features = sampleFromComponent(components[componentIdx]);
    return { componentIdx, features };
  };

  const gmm = initializeGMM(GMM_COMPONENTS, FEATURE_DIMENSION);

  const initializeDataPoints = () => {
    // Pre-populate the CPU-side buffer directly
    const view = new DataView(dataPoints._buffer);

    for (let i = 0; i < DATA_POINT_COUNT; i++) {
      const base = i * dataPoints.byteSize;

      // id: u32
      view.setUint32(base + dataPoints.offsets.id, i, true);

      // position: vec2<f32>
      view.setFloat32(base + dataPoints.offsets.position, Math.random() * size.width, true);
      view.setFloat32(base + dataPoints.offsets.position + 4, Math.random() * size.height, true);

      // orientation: vec2<f32>
      const angle = Math.random() * 2 * Math.PI;
      view.setFloat32(base + dataPoints.offsets.orientation, Math.cos(angle), true);
      view.setFloat32(base + dataPoints.offsets.orientation + 4, Math.sin(angle), true);

      // features: array<f32, FEATURE_DIMENSION> sampled from GMM
      const { componentIdx, features } = sampleFromGMM(gmm);
      for (let j = 0; j < FEATURE_DIMENSION; j++) {
        view.setFloat32(base + dataPoints.offsets.features + j * 4, features[j], true);
      }

      // label: u32 (which component the feature was drawn from)
      view.setUint32(base + dataPoints.offsets.label, componentIdx, true);
    }

    // Single GPU write after populating entire buffer
    device.queue.writeBuffer(dataPoints._gpubuffer, 0, dataPoints._buffer);
  };
  initializeDataPoints();

  // compute pass - physics simulation
  function computePass() {
    // Update textures
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();

      pass.setBindGroup(pipeline.index, pipeline.bindGroup);
      pass.setPipeline(update_textures);
      pass.dispatchWorkgroups(...WORKGROUP_COUNT_TEXTURE);

      pass.end();
      device.queue.submit([encoder.finish()]);
    }

    // Update positions
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();

      pass.setBindGroup(pipeline.index, pipeline.bindGroup);
      pass.setPipeline(update_positions);
      pass.dispatchWorkgroups(WORKGROUP_COUNT_BUFFER);

      pass.end();
      device.queue.submit([encoder.finish()]);
    }
  }

  const gui = new GUI();
  gui.add({ reset: () => initializeDataPoints() }, "reset");

  controls.compute_steps = 200;
  gui.add(controls, "compute_steps").min(1).max(200).step(1).name("Compute Steps");

  controls.sensor_angle = Math.PI / 2;
  gui.add(controls, "sensor_angle").min(0.01).max(Math.PI).name("Sensor Angle");

  controls.sensor_offset = 20;
  gui.add(controls, "sensor_offset").min(2).max(50).name("Sensor Offset");

  controls.steer_angle = Math.PI / 2;
  gui.add(controls, "steer_angle").min(0.01).max(Math.PI).name("Steer Angle");

  controls.persistence = 5.0;
  gui.add(controls, "persistence").min(0).max(5.0).step(0.01).name("Trail Persistence");

  controls.visualization_mode = 0;
  gui.add(controls, "visualization_mode", { Label: 0, Density: 1 }).name("Visualization");

  controls.density_repulsion_strength = 0.5;
  gui.add(controls, "density_repulsion_strength").min(0).max(2.0).step(0.1).name("Density Repulsion");

  function frame() {
    for (let i = 0; i < controls.compute_steps; i++) {
      canvas.key = [canvas.key[0] + 1, canvas.key[1]];
      computePass();
    }
    renderPass(device, context, render, pipeline.bindGroup, pipeline.index);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch(console.error);
