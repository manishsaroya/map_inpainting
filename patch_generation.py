import pickle
with open('synthetic_dataset.pickle','rb') as f:
	data = pickle.load(f)

#print(data,len(data),type(data))
PATCH_SIZE =  24
train_x = data['training_data']
train_y = data['training_labels']
test_x = data['testing_data']
test_y = data['testing_labels']

import numpy as np
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
def get_patch(blur_patch,image,direction,start_pt):
	im = np.ones(image.shape)
	#print(start_pt)
	rc = range(start_pt,start_pt+PATCH_SIZE)
	r = range(image.shape[0]-PATCH_SIZE,image.shape[0])
	c = range(image.shape[1]-PATCH_SIZE,image.shape[1])
	x = range(0,PATCH_SIZE) 
	if direction == 0:
		before = np.sum(im[np.ix_(x,rc)] )
		im[np.ix_(x,rc)] = blur_patch
		after = np.sum(im[np.ix_(x,rc)] )
	elif direction == 1:
		before = np.sum(im[np.ix_(r,rc)])
		im[np.ix_(r,rc)] = blur_patch
		after = np.sum(im[np.ix_(r,rc)] )
	elif direction == 2:
		before = np.sum(im[np.ix_(rc,x)])
		im[np.ix_(rc,x)] = blur_patch
		after = np.sum(im[np.ix_(rc,x)])
	elif direction == 3:
		before = np.sum(im[np.ix_(rc,c)])
		im[np.ix_(rc,c)] = blur_patch
		after = np.sum(im[np.ix_(rc,c)])
	print(im.shape)
	print("if path exists",np.sum(image[np.ix_(rc,c)]))
	print("before",before,"after",after,"total before",im.shape[0]*im.shape[1],"total after",np.sum(im))
	return im	
def get_patched_image(image):

	dr = np.random.randint(4,size=1)
	print("image is ",image)
	print("number of ones is ", np.sum(image))
	start_pt = np.random.randint(image.shape[0]-PATCH_SIZE+1,size=1)
	blur_patch = np.zeros(shape=(PATCH_SIZE,PATCH_SIZE))
	mask = get_patch(blur_patch,image,dr[0],start_pt[0])
	image = np.minimum(image,mask)
	print("direction chosen",dr[0],"starting point is ",start_pt[0])
	print("masked image",image)
	print("sum is ",np.sum(image))

	return image

#print(tiles)
import matplotlib.pyplot as plt
# plt.imshow(train_x[0])
# plt.ylabel('y')
# plt.xlabel('x')
# plt.show()
image = np.float32(get_patched_image(train_x[1]))
fig = plt.figure(figsize=(10,10))
fig.add_subplot(1,2,1)
plt.imshow(train_x[1])
plt.title("True image")
plt.ylabel('y')
plt.xlabel('x')
fig.add_subplot(1,2,2)
plt.imshow(image)
title = "Masked for PATCH_SIZE = " + str(PATCH_SIZE)
plt.title(title)
plt.ylabel('masked_y')
plt.xlabel('masked_x')
plt.show()
plt.savefig("figure_8.png")
# x = np.zeros(shape=(4,4))
# x[np.ix_([1,2],[1,2])]= np.ones((2,2))
# print(x)