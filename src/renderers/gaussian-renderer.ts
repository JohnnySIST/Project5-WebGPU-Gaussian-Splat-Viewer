import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setGaussianMultiplier: (multiplier: number) => void,
}

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  sh_buffer: GPUBuffer,
): GaussianRenderer {

  const  sorter = get_sorter(pc.num_points, device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nullData = new Uint32Array([0]);
  const nullDataBuffer = device.createBuffer({
    label: 'null data buffer',
    size: nullData.byteLength,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(nullDataBuffer.getMappedRange()).set(nullData);
  nullDataBuffer.unmap();

  const indirectData = new Uint32Array([6, pc.num_points, 0, 0]);
  const indirectDrawBuffer = device.createBuffer({
    label: 'indirect draw buffer',
    size: indirectData.byteLength,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(indirectDrawBuffer.getMappedRange()).set(indirectData);
  indirectDrawBuffer.unmap();

  const renderSettingsBuffer = device.createBuffer({
    label: 'render settings',
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const splatBuffer = device.createBuffer({
    label: 'splat buffer',
    size: pc.num_points * 32 * 10,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // ===============================================
  //    Create Bind Group Layouts
  // ===============================================
  
  const camera_bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
    ]
  });
  
  const gaussian_bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ]
  });

  const sort_bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]
  });

  const sort_bind_group_layout_vertex = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ]
  });

  const render_settings_bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }
    ]
  });

  const camera_settings_bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: "uniform" } }
    ]
  });

  const splat_bind_group_layout_compute = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ]
  });
  
  const splat_bind_group_layout_vertex = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }
    ]
  });

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================

  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: device.createPipelineLayout({
      bindGroupLayouts: [
        camera_settings_bind_group_layout,
        gaussian_bind_group_layout,
        sort_bind_group_layout,
        splat_bind_group_layout_compute,
      ]
    }),
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: sort_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  const sort_bind_group_vertex = device.createBindGroup({
    label: 'sort vertex',
    layout: sort_bind_group_layout_vertex,
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  
  const render_shader = device.createShaderModule({code: renderWGSL});
  const render_pipeline = device.createRenderPipeline({
    label: 'render',
    layout: device.createPipelineLayout({
      bindGroupLayouts: [
        camera_settings_bind_group_layout,
        gaussian_bind_group_layout,
        sort_bind_group_layout_vertex,
        splat_bind_group_layout_vertex,
      ]
    }),
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        },
      }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'gaussian camera',
    layout: camera_bind_group_layout,
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'gaussian data',
    layout: gaussian_bind_group_layout,
    entries: [{ binding: 0, resource: { buffer: pc.gaussian_3d_buffer } }],
  });

  const settings_bind_group = device.createBindGroup({
    label: 'render settings',
    layout: render_settings_bind_group_layout,
    entries: [{ binding: 0, resource: { buffer: renderSettingsBuffer } }],
  });

  const camera_settings_bind_group = device.createBindGroup({
    label: 'camera & settings',
    layout: camera_settings_bind_group_layout,
    entries: [{ binding: 0, resource: { buffer: camera_buffer } },
              { binding: 1, resource: { buffer: renderSettingsBuffer } }],
  });

  const splat_bind_group_compute = device.createBindGroup({
    label: 'splat data compute',
    layout: splat_bind_group_layout_compute,
    entries: [{ binding: 0, resource: { buffer: splatBuffer },},
              { binding: 1, resource: { buffer: sh_buffer }}],
  });

  const splat_bind_group_render = device.createBindGroup({
    label: 'splat data vertex',
    layout: splat_bind_group_layout_vertex,
    entries: [{ binding: 0, resource: { buffer: splatBuffer } }],
  });

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    encoder.copyBufferToBuffer(nullDataBuffer, 0, sorter.sort_info_buffer, 0, 4);
    encoder.copyBufferToBuffer(nullDataBuffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);

    const computePass = encoder.beginComputePass(
      { label: 'preprocess pass' }
    );
    computePass.setPipeline(preprocess_pipeline);
    // maximun number of binding groups is 4, so merging uniforms
    computePass.setBindGroup(0, camera_settings_bind_group);
    computePass.setBindGroup(1, gaussian_bind_group);
    computePass.setBindGroup(2, sort_bind_group);
    computePass.setBindGroup(3, splat_bind_group_compute);
    computePass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    computePass.end();

    // debug: check sort_infos.keys_size
    const keysizeCheckBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, keysizeCheckBuffer, 0, 4);
    device.queue.onSubmittedWorkDone().then(async () => {
      await keysizeCheckBuffer.mapAsync(GPUMapMode.READ);
      const array = new Uint32Array(keysizeCheckBuffer.getMappedRange());
      console.log('keysize:', array[0]);
      keysizeCheckBuffer.unmap();
    });

    // sorting
    sorter.sort(encoder);

    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirectDrawBuffer, 4, 4);
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_settings_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group_vertex);
    pass.setBindGroup(3, splat_bind_group_render);
    pass.drawIndirect(indirectDrawBuffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      render(encoder, texture_view);
    },
    camera_buffer,
    setGaussianMultiplier: (multiplier: number) => {
      device.queue.writeBuffer(renderSettingsBuffer, 0, new Float32Array([multiplier, pc.sh_deg]) );
    },
  };
}
