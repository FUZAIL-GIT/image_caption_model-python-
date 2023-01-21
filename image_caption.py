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
mDelta  = mEnd - mStart
print("Device is "+ device.type)
print("Model loaded in %.2f seconds" % mDelta)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
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
    #print("temperature: ", temperature)

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

#folder_dir = "Images"
#file_name = "horse-7698761_960_720.webp"
#print("File name: "+file_name)
#start = time.time()
#print(predict_step([folder_dir+"/"+file_name]))
#pass
#end = time.time()
#delta = end - start
#deltaMin = delta/60
#print("Execution time: %.2f seconds." % delta)


#####Start of Pic2Story#####
folder_dir = "images"
file_name = "1.jpg"
print("File name: "+file_name)
start = time.time()
prompt = predict_step([folder_dir+'/'+file_name])
print("Image Caption: " + prompt[0])
end = time.time()
delta = end - start
print("*****Image Caption generation time: %.2f seconds.*****" % delta)
#tell a blind person in detail "a white horse standing next to a person" as you are seeing and telling to a blind person in positive way
start = time.time()
print("Detail view description to visually impaired person: "+openai_create('tell a blind person in detail "'+prompt[0]+'" as you are seeing and telling to a blind person in positive way\n', "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", 0.5))
end = time.time()
delta = end - start
print("*****Description generated in %.2f seconds.*****" % delta)
#########End of Pic2Story############