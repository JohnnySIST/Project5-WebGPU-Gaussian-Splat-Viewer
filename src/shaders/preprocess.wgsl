const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
    fov: vec2<f32>,
};

struct RenderSettings {
    gaussian_multiplier: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    position: vec2<f32>,
    radius: f32,
    color: vec4<f32>,
    conic: vec3<f32>,
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings : RenderSettings;
@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;
@group(3) @binding(0)
var<storage,read_write> splats : array<Splat>;
@group(3) @binding(1)
var<storage,read> sh_coeffs : array<u32>;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    if (c_idx % 2u == 0u) {
        let a = unpack2x16float(sh_coeffs[splat_idx * 24u + c_idx / 2u * 3u]);
        let b = unpack2x16float(sh_coeffs[splat_idx * 24u + c_idx / 2u * 3u + 1u]);
        return vec3<f32>(a.x, a.y, b.x);
    } else {
        let a = unpack2x16float(sh_coeffs[splat_idx * 24u + c_idx / 2u * 3u + 1u]);
        let b = unpack2x16float(sh_coeffs[splat_idx * 24u + c_idx / 2u * 3u + 2u]);
        return vec3<f32>(a.y, b.x, b.y);
    }
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

// from https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L216
fn computeCov2D(mean: vec3<f32> , focal_x: f32, focal_y: f32, tan_fovx: f32, tan_fovy: f32, cov3D: mat3x3<f32>) -> mat2x2<f32> {
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	var t = camera.view * vec4<f32>(mean, 1.0);

	let limx = 1.3f * tan_fovx;
	let limy = 1.3f * tan_fovy;
	let txtz = t.x / t.z;
	let tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	let J = mat3x3<f32>(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	let W = mat3x3<f32>(
        camera.view[0][0], camera.view[0][1], camera.view[0][2],
        camera.view[1][0], camera.view[1][1], camera.view[1][2],
        camera.view[2][0], camera.view[2][1], camera.view[2][2]
    );
    let T = W * J;

	var cov = transpose(T) * transpose(cov3D) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return mat2x2<f32>(cov[0][0], cov[0][1], cov[1][0], cov[1][1]);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction

    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let c = unpack2x16float(vertex.rot[0]);
    let d = unpack2x16float(vertex.rot[1]);
    let e = unpack2x16float(vertex.scale[0]);
    let f = unpack2x16float(vertex.scale[1]);
    let world_pos = vec3<f32>(a.x, a.y, b.x);
    let opacity = 1.0/(1.0+exp(-b.y)); // decode opacity from sigmoid space
    let rot = vec4<f32>(c.x, c.y, d.x, d.y);
    let scale = exp(vec3<f32>(e.x, e.y, f.x)); // scale is stored in log space
    var ndc_pos = camera.proj * camera.view * vec4<f32>(world_pos, 1.0);
    ndc_pos /= ndc_pos.w;

    if (ndc_pos.x > -1.2 && ndc_pos.x < 1.2 && ndc_pos.y > -1.2 && ndc_pos.y < 1.2 && ndc_pos.z > 0.0 && ndc_pos.z < 1.0) {
        let qr = rot.x;
        let qx = rot.y;
        let qy = rot.z;
        let qz = rot.w;
        let R = mat3x3<f32>(
            1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qr * qz), 2.0 * (qx * qz + qr * qy),
            2.0 * (qx * qy + qr * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qr * qx),
            2.0 * (qx * qz - qr * qy), 2.0 * (qy * qz + qr * qx), 1.0 - 2.0 * (qx * qx + qy * qy)
        );
        let S = mat3x3<f32>(
            vec3<f32>(render_settings.gaussian_multiplier * scale.x, 0.0, 0.0),
            vec3<f32>(0.0, render_settings.gaussian_multiplier * scale.y, 0.0),
            vec3<f32>(0.0, 0.0, render_settings.gaussian_multiplier * scale.z)
        );
        let cov3D = transpose(S * R) * S * R;

        let cov2D = computeCov2D(world_pos, camera.focal.x, camera.focal.y, tan(camera.fov.x), tan(camera.fov.y), cov3D);

        // from https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L216
        // Invert covariance (EWA algorithm)
	    let det = (cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[1][0]);
	    if (det == 0.0f) {
	    	return;
	    }
	    let det_inv = 1.f / det;
        let conic = mat2x2<f32>(
            cov2D[1][1] * det_inv, -cov2D[0][1] * det_inv,
            -cov2D[1][0] * det_inv, cov2D[0][0] * det_inv
        );

	    // Compute extent in screen space (by finding eigenvalues of
	    // 2D covariance matrix). Use extent to compute a bounding rectangle
	    // of screen-space tiles that this Gaussian overlaps with. Quit if
	    // rectangle covers 0 tiles.
	    let mid = 0.5f * (cov2D[0][0] + cov2D[1][1]);
	    let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	    let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	    let radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

        // compute color from SH
        let cam_pos = -camera.view[3].xyz;
        let dir = normalize(world_pos - cam_pos);
        let color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));

        let splat_idx = atomicAdd(&sort_infos.keys_size, 1u);
        splats[splat_idx].position = ndc_pos.xy;
        splats[splat_idx].radius = radius;
        splats[splat_idx].color = vec4<f32>(color, opacity);
        splats[splat_idx].conic = vec3<f32>(conic[0][0], conic[0][1], conic[1][1]);

        sort_depths[splat_idx] = u32(clamp(1.0-ndc_pos.z, 0.0, 1.0) * f32(0xFFFFFFFFu));
        sort_indices[splat_idx] = splat_idx;
        let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
        // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
        if (splat_idx % keys_per_dispatch == 0u) {
            atomicAdd(&sort_dispatch.dispatch_x, 1u);
        }
    }
}