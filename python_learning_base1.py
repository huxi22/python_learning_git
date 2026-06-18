import torch

learning_rate = 0.01
print(learning_rate)

scores = [1, 2, 3]
print(scores[0])

scores.append(5)
print(scores[3])

model_config = {
   "input_size": 10,
   "hidden_size": 20, 
   "num_classes": 5
}
print(model_config["input_size"])

image_size = (28, 20)
height, width = image_size
print(height)
print(width)


if image_size[0]<30:
    print("height of image size is less than 30")
else:
    print("height of image size is greater than or equal to 30")

def image_add(image_size):
    return image_size[0] + image_size[1]

print(image_add(image_size))






