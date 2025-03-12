import gradio as gr
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    """Generates a caption for an uploaded image."""
    raw_image = Image.fromarray(input_image).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption, None, None  # Ensure it returns three values

def caption_url_images(url: str):
    """Extracts image URLs from a webpage and generates captions for them."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_elements = soup.find_all('img')
    
    captions = []
    
    for img_element in img_elements:
        img_url = img_element.get('src')
        if not img_url:
            continue
        
        if 'svg' in img_url or '1x1' in img_url:
            continue
        
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http'):
            continue
        
        try:
            img_response = requests.get(img_url)
            raw_image = Image.open(BytesIO(img_response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:
                continue
            
            raw_image = raw_image.convert('RGB')
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(f"{img_url}: {caption}")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue
    
    if captions:
        file_path = "captions.txt"
        with open(file_path, "w") as file:
            file.write("\n".join(captions))
        return None, "Captions generated successfully! Download the file below.", file_path  # Return three values
    else:
        return None, "No captions generated.", None  # Return three values

def process_input(input_type, image, url):
    """Determines whether to process an image or a URL based on user selection."""
    if input_type == "Upload Image":
        return caption_image(image)
    elif input_type == "Enter URL":
        return caption_url_images(url)
    return None, "Please select a valid input type.", None  # Ensure three outputs for safety

iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Radio(["Upload Image", "Enter URL"], label="Select Input Type"),
        gr.Image(label="Upload Image", type="numpy", interactive=True),
        gr.Textbox(label="Enter URL", interactive=True)
    ],
    outputs=[
        gr.Textbox(label="Generated Caption for Image"),
        gr.Markdown("Generated Caption for the input as URL"),
        gr.File(label="Download Captions File")
    ],
    title="Image Captioning Application",
    description="Upload an image to get a caption or enter a URL to extract and caption images from a webpage."
)

iface.launch()
