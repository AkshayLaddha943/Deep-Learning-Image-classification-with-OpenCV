

import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    
rows = open('D:\\Python Projects\\OpenCV\\Deep Learning Image Classification - GoogleNet\\synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]

net = cv2.dnn.readNetFromCaffe("D:\\Python Projects\\OpenCV\\Deep Learning Image Classification - GoogleNet\\bvlc_googlenet.prototxt", "D:\\Python Projects\\OpenCV\\Deep Learning Image Classification - GoogleNet\\bvlc_googlenet.caffemodel")

image = cv2.imread("D:\\Python Projects\\OpenCV\\Deep Learning Image Classification - GoogleNet\\ostrich.jpg")

blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
print(blob.shape)

net.setInput(blob)
preds = net.forward()

t, _ = net.getPerfProfile()
print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

#10 indexes with highest probability
indexes = np.argsort(preds[0])[::-1][:10]

text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]], preds[0][indexes[0]] * 100)
y0, dy = 30, 30
for i, line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
# Print top 10 prediction:
for (index, idx) in enumerate(indexes):
    print("{}. label: {}, probability: {:.10}".format(index + 1, classes[idx], preds[0][idx]))
    
fig = plt.figure(figsize=(10, 6))
plt.suptitle("Image classification with OpenCV using GoogLeNet and caffe pre-trained models", fontsize=14,
             fontweight='bold')
fig.patch.set_facecolor('silver')

show_img_with_matplotlib(image, "GoogLeNet and caffe pre-trained models", 1)

# Show the Figure:
plt.show()