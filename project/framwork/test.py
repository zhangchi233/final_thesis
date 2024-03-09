from diffusers import StableDiffusionLDM3DPipeline

pipe = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-pano")

# On CPU


# On GPU
pipe.to("cuda")

prompt = "360 view of a large bedroom"
name = "lemons"

output = pipe(prompt)
rgb_image, depth_image = output.rgb, output.depth
rgb_image[0].save(name+"_ldm3d_rgb.jpg")
depth_image[0].save(name+"_ldm3d_depth.png")