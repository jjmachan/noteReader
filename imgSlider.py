import time
import cv2
from sys import argv
import imutils
import label_image as li
import tensorflow as tf
import numpy as np

xrange = range
file_name = "./note-eighth-c1-870.jpg"
model_file = "./out_mobnet_100_96_1oct_Copy/output_graph.pb"
label_file = "./out_mobnet_100_96_1oct_Copy/output_labels.txt"
input_height = 96
input_width = 96
input_mean = 0
input_std = 255
input_layer = "Placeholder"
output_layer = "final_result"

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		print(w)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], windowSize[0]):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


image = cv2.imread(argv[1],1)
# b = -30. # brightness
# c = 190.  # contrast
# img = cv2.addWeighted(img, 1. + c/127., img, 0, b-c)
#print(image[3,3])
(winW, winH) = (96,96)

#load tensorflow pretrained model



for (i, resized) in enumerate(pyramid(image, scale=2)):
	cv2.imshow("Layer {}".format(i + 1), resized)
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(image, stepSize=40, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		clone = image.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		print(x,y)
		croped = clone[y:y+winH, x:x+winW]
		#cv2.imshow("croped image", croped)
		# img2= cv2.resize(croped,dsize=(64,16,3), interpolation = cv2.INTER_CUBIC)
		# np_image_data = np.asarray(img2)
		# np_final = np.expand_dims(np_image_data,axis=0)



		#print(np_final.shape)

		graph = li.load_graph(model_file)
		# t = li.read_tensor_from_image_file(
		# 	file_name,
		# 	input_height=input_height,
		# 	input_width=input_width,
		# 	input_mean=input_mean,
		# 	input_std=input_std
		# 	)
		print(croped.shape)
		cv2.imshow("croped image", croped)
		t = croped
		#t = np.pad(croped,((0,0),(38,38),(0,0)),'constant')
		# t = t.reshape(1,224,224,3)
		t = t.reshape(1,96,96,3)
		print(t.shape)

		input_name = "import/" + input_layer
		output_name = "import/" + output_layer
		input_operation = graph.get_operation_by_name(input_name)
		output_operation = graph.get_operation_by_name(output_name)

		with tf.Session(graph=graph) as sess:
			results = sess.run(output_operation.outputs[0], {
			    input_operation.outputs[0]: t
			})
		results = np.squeeze(results)

		top_k = results.argsort()[-5:][::-1]
		labels = li.load_labels(label_file)
		for i in top_k:
			print(labels[i], results[i])

		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.25)
cv2.destroyAllWindows()

