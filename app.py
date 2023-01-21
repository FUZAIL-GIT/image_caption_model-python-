# from flask import Flask, render_template, request
# import os
# import openai
# from os import listdir
# from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
# import torch
# from PIL import Image
# import time
# mStart = time.time()
# model = VisionEncoderDecoderModel.from_pretrained("vit-gpt2-image-captioning")
# feature_extractor = ViTFeatureExtractor.from_pretrained("vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("vit-gpt2-image-captioning")
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model.to(device)
# mEnd = time.time()
# mDelta  = mEnd - mStart
# print("Device is "+ device.type)
# print("Model loaded in %.2f seconds" % mDelta)
#
# max_length = 16
# num_beams = 4
# gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
#
# app = Flask(__name__)
#
# @app.route("/") # at the end point /
# def hello(): # call method hello
#     return 'Hello World!' # which returns “hello world”
#
# @app.route('/') # at the end point /
# def hello_name(name): # call method hello_name
#     return 'Hello '+ name # which returns “hello + name
#
# if __name__ == '__main__': # on running python app.py
#     app.run(debug=True) # run the flask app
#
# @app.route('/uploader', methods = ['GET', 'POST'])
# def predict_step(image_paths):
#   if request.method == 'POST' :
#   f =request.files['files']
#   images = []
#   print(f)
#   for image_path in request.files:
#     i_image = Image.open(image_path)
#     if i_image.mode != "RGB":
#       i_image = i_image.convert(mode="RGB")
#
#     images.append(i_image)
#
#   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#   pixel_values = pixel_values.to(device)
#
#   output_ids = model.generate(pixel_values, **gen_kwargs)
#
#   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#   preds = [pred.strip() for pred in preds]
#
#   prompt =preds
#   return 'Hello World'
#       # openai_create('tell a blind person in detail "'+prompt[0]+'" as you are seeing and telling to a blind person in positive way\n', "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", 0.5)
#
#
#
# def openai_create(prompt, openai_api_key, temperature):
#     #print("temperature: ", temperature)
#     print(prompt)
#     openai.api_key = openai_api_key
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=prompt,
#         temperature=temperature,
#         max_tokens=2000,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     return response.choices[0].text