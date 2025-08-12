import gradio as gr
from diffusers import DiffusionPipeline
import torch

# Load the Stable Diffusion model
pipeline = DiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5',
                                             torch_dtype=torch.float32,
                                             use_safetensors=True)

# Move the pipeline to the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)

# Define the Gradio interface
def generate_image(prompt, negative_prompt):
    generate_images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        guidance_scale=7.5,
        num_inference_steps=35
    ).images
    return generate_images[0]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Image Generation")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your image description here...")
            negative_prompt_input = gr.Textbox(label="Negative Prompt", placeholder="What you don't want in the image...")
            generate_button = gr.Button("Generate Image")
        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    generate_button.click(fn=generate_image, inputs=[prompt_input, negative_prompt_input], outputs=output_image)

# Run the interface
demo.launch(share=True)