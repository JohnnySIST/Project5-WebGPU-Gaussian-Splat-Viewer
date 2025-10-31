struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
    fov: vec2<f32>,
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct RenderSettings {
    gaussian_multiplier: f32,
    sh_deg: f32,
}

struct Splat {
    //TODO: store information for 2D splat rendering
    position: vec2<f32>,
    radius: f32,
    color: vec4<f32>,
    conic: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings : RenderSettings;
@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(2) @binding(2)
var<storage,read> sort_indices : array<u32>;
@group(3) @binding(0)
var<storage,read> splats : array<Splat>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>,
    @location(1) center: vec2<f32>,
    @location(2) conic: vec3<f32>,
};

// strww

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    let splat_idx = sort_indices[instance_index];

    var out: VertexOutput;
    let quads = array<vec2<f32>,6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );

    let quad_corner = quads[vertex_index];
    let splat = splats[splat_idx];
    let offset = vec2<f32>(quad_corner.x / camera.viewport.x, quad_corner.y / camera.viewport.y) * splat.radius;
    out.position = vec4<f32>(splat.position + offset, 0.0, 1.0);
    out.center = splat.position;
    out.color = splat.color;
    out.conic = splat.conic;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let splat_center = (vec2<f32>(in.center.x, -in.center.y) + vec2<f32>(1.0, 1.0)) / 2.0 * camera.viewport;
    let dis = in.position.xy - splat_center;
    let value = in.conic.x * dis.x * dis.x + in.conic.y * dis.x * dis.y + in.conic.z * dis.y * dis.y;
    if (value < 0.0 || value > 10.0) {
        discard;
    }
    let alpha = in.color.a * exp(-0.5 * value);
    return vec4<f32>(in.color.rgb, alpha);
}