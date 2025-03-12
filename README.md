# Image Captioning Application

## Project Overview
The **Image Captioning Application** is a Python-based application that utilizes deep learning to generate captions for images. This project leverages **Gradio** for a user-friendly interface, **Hugging Face Transformers** for image captioning, and **BeautifulSoup** for web scraping images from a URL. Users can either upload an image or provide a webpage URL containing images, and the tool generates descriptive captions.

## Introduction
Automatic image captioning is an essential task in computer vision and natural language processing. It is widely used in accessibility applications, content management, and AI-driven storytelling. This project implements **Salesforce's BLIP (Bootstrapped Language-Image Pretraining) model** for image captioning and provides an easy-to-use web interface using **Gradio**.


<p align="center">
  <img src="https://github.com/so123-design/Image-Captioning-Application/blob/4de7793ad772ab28e6e4d30a0b02bd741f278da9/Image%20captioning%202.PNG" alt="My Image" width="800">
</p>


## Features
- **Upload an image** to receive an AI-generated caption.
- **Provide a webpage URL**, and the tool will extract images and generate captions for them.
- **Downloadable text file** containing all captions for URL-extracted images.
- **User-friendly Gradio interface** for seamless interaction.

## Technologies Used
- **Python**: Core programming language.
- **Gradio**: For building an interactive web-based UI.
- **Hugging Face Transformers**: For leveraging the BLIP image captioning model.
- **BeautifulSoup**: For extracting images from a given webpage.
- **PIL (Pillow)**: For image processing.
- **NumPy**: For handling image arrays.
- **Requests**: For fetching webpage data and images.


```

## Installation and Setup
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Install Dependencies
```sh
pip install gradio transformers pillow requests beautifulsoup4 numpy torch
```

### Run the Application
```sh
python main.py
```

## Code Explanation
### 1. Loading the Pretrained Model
```python
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```
This loads the **BLIP image captioning model** and its processor from Hugging Face.

### 2. Captioning an Uploaded Image
```python
def caption_image(input_image: np.ndarray):
    raw_image = Image.fromarray(input_image).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True), None
```
This function takes an uploaded image, processes it, and generates a caption.

### 3. Captioning Images from a Webpage
```python
def caption_url_images(url: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_elements = soup.find_all('img')
    
    captions = []
    for img_element in img_elements:
        img_url = img_element.get('src')
        if img_url and img_url.startswith(('http', '//')):
            img_url = 'https:' + img_url if img_url.startswith('//') else img_url
            
            try:
                img_response = requests.get(img_url)
                raw_image = Image.open(BytesIO(img_response.content)).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt")
                out = model.generate(**inputs, max_length=50)
                captions.append(f"{img_url}: {processor.decode(out[0], skip_special_tokens=True)}")
            except Exception as e:
                print(f"Error processing {img_url}: {e}")
    
    if captions:
        file_path = "captions.txt"
        with open(file_path, "w") as file:
            file.write("\n".join(captions))
        return None, file_path
    return "No captions generated.", None
```
This function extracts images from a webpage, downloads them, processes them through the model, and saves the generated captions.

### 4. Processing User Input
```python
def process_input(input_type, image, url):
    if input_type == "Upload Image":
        return caption_image(image)
    elif input_type == "Enter URL":
        return caption_url_images(url)
```
This function determines whether to process an image upload or extract captions from a URL.

### 5. Gradio Interface
```python
import gradio as gr

iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Radio(["Upload Image", "Enter URL"], label="Select Input Type"),
        gr.Image(label="Upload Image", type="numpy", interactive=True),
        gr.Textbox(label="Enter URL", interactive=True)
    ],
    outputs=[
        gr.Textbox(label="Generated Caption for the input as Image"),
        gr.Markdown("Generated Caption for the input as URL"),
        gr.File(label="Download Captions File")
    ],
    title="Image Captioning Application",
    description="Upload an image to get a caption or enter a URL to extract and caption images from a webpage."
)

iface.launch()
```
This sets up a **Gradio web interface** for users to interact with the model easily.

## Conclusion
This project demonstrates the power of **deep learning and natural language processing** in **image captioning**. By integrating **Gradio**, it provides an intuitive interface, making AI-powered caption generation accessible to users. The project can be extended with **better image filtering, multi-language support, or enhanced deep learning models** for improved accuracy.
