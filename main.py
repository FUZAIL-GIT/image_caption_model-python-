from flask import Flask, request, jsonify
import os
import openai
from os import listdir
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import time

mStart = time.time()
model = VisionEncoderDecoderModel.from_pretrained("vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
mEnd = time.time()
mDelta = mEnd - mStart
print("Device is " + device.type)
print("Model loaded in %.2f seconds" % mDelta)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
app = Flask(__name__)


@app.route('/caption_description', methods=['POST'])
def caption_description():
    if 'image' not in request.files:
        return jsonify(error='Image file is missing'), 400
    image_file = request.files['image']
    image_path = save_image_to_folder(image_file)
    caption = predict_step([image_path])[0]
    description = openai_create(
        'tell a blind person in detail "' + caption + '" as you are seeing and telling to a blind person in positive way\n',
        "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", 0.5)
    return jsonify(caption=caption, description=description)


def save_image_to_folder(image_file):
    image_path = 'images/' + image_file.filename
    image_file.save(image_path)
    return image_path


def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def openai_create(prompt, openai_api_key, temperature):
    # print("temperature: ", temperature)

    openai.api_key = openai_api_key
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=temperature,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text


if __name__ == '__main__':
    # replace YOU IP ADDRESS
    app.run(ssl_context='adhoc', host="192.168.0.100")
