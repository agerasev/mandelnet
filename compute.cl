float abs2(float2 z) {
    return z.x*z.x + z.y*z.y;
}

__kernel void compute(
    __write_only image2d_t depth_img,
    float2 pos, float zoom,
    int max_depth,
    int julia, float2 jz
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float2 c = pos + zoom*(convert_float2(p) - 0.5f*convert_float2(s))/min(s.x, s.y);
    float2 z = julia ? jz : c;
    
    float d = -1;
    int i;
    for (i = 0; i < max_depth; ++i) {
        if (abs2(z) > (1<<16)) {
            d = i + 1 - log(log(abs2(z))/(2*log(2.0f)))/log(2.0f);
            break;
        }
        z = (float2)(z.x*z.x - z.y*z.y, 2.0f*z.x*z.y) + c;
    }
    
    write_imagef(depth_img, p, d);
}

__kernel void colorize(
    __read_only image2d_t depth_img,
    __write_only image2d_t color_img,
    int samples,
    __read_only image1d_t color_map,
    float map_period
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
    
    int ix, iy;
    float4 color = (float4)(0,0,0,0);
    for (iy = 0; iy < samples; ++iy) {
        for (ix = 0; ix < samples; ++ix) {
            float depth = read_imagef(depth_img, samples*p + (int2)(ix, iy)).x;
            if (depth >= 0.0f) {
                color += (float4)read_imagef(color_map, sampler, depth/map_period);
            } else {
                color += (float4)(0,0,0,1);
            }
        }
    }
    color /= samples*samples;
    
    write_imagef(color_img, p, color);
}