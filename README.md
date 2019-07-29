# **Traffic Sign Recognition** 

**The goals / steps of this project are the following:** 
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plot/training_set_counts.png "Visualization"
[image2]: ./plot/random_examples.png "Random Examples"
[image3]: ./plot/image_transformations.png "Transformations"
[image4]: ./plot/augmented_img_example.png "Augmentation"
[image5]: ./plot/learning_curve.png "Learning Curve"
[image6]: ./plot/new_images.png "New Images"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

## Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is 
ditributed across the different labels.

![alt text][image1]

The average number of training examples per class is 809, the minimum is 180 and the maximum 2010, 
hence some labels are one order of magnitude more abundant than others.

Most common signs:
* `Speed limit (50km/h)`  train samples: 2010
* `Speed limit (30km/h)`  train samples: 1980
* `Yield`  train samples: 1920
* `Priority road`  train samples: 1890
* `Keep right`  train samples: 1860


Most rare signs:

* `Speed limit (20km/h)`  train samples: 180
* `Dangerous curve to the left`  train samples: 180
* `Go straight or left`  train samples: 180
* `Pedestrians`  train samples: 210
* `End of all speed and passing limits`  train samples: 210

Here is an visualization of some 10 randomly picked training examples for each class. 
As we can see, within each class there is a high variability in appearance due to different 
weather conditions, time of the day and image angle.

![alt text][image2]

---

## Design and Test a Model Architecture

#### Question 1:
*Describe how you preprocessed the data. Why did you choose that technique?*
##### Answer
Following a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) 
I applied similar normalization and image enhancements . Images were transformed in the YUV space and adjusted by 
histogram sketching and by increasing sharpness.  Finally only the Y channel was selected as in some preliminary 
experiments full color images seem to confuse the classifier (as also reported in the published baseline), 
the latter effect however may depend on the network architecture, as in the long term we would intuitively expect 
to have networks trained with full color images to perform better.

Here is an example of an original image and the transformed image.

![alt text][image3]

Hence the difference between the original data set and the augmented data set is the following is the reduced level of noise 
and number of channels.


#### Question 2:
*Describe how you set up the training, validation and testing data for your model. 
Optional: If you generated additional data, how did you generate the data? Why did you generate the data? 
What are the differences in the new dataset (with generated data) from the original dataset?*
##### Answer:

All images were processed by transform_img function as discribed in the question 1. 
Training test and validation set were provided in the exercise. 
Training set was also augmented by generating 5 additional images from every given image.
Images were augmented by `augment_img function`. The process consists of random rotation around image center 
(random value between -15 and 15 deg) and random vertical stretching (as the simplest way to simulate different 
viewing angle) by random value up to `40 %`.

An example of an image aftern augmentation is shown below:

![alt text][image4]

#### Question 3:

*Describe what your final model architecture looks like.*

##### Answer
My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Y channel image   				    |
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Relu					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 14x14x6    |
| Relu					|												|
| Fully connected		| Input 14x14x6 = 400  output 120     			|
| Relu					|												|
| droupout      		|   							                |
| Fully connected		| Input 120  output 84     						|
| Relu					|												|
| dropuout				|   									        |
| Fully connected		| Input 84  output 43     						|
| Softmax				|												|
|						|												|

#### Question 4:
*How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)*

##### Answer:

I trained the model using an Adam optimizer ,  learning rate of 1e-4 , dropout rate of 0.3 and batch size of 128.


#### Question 5
*What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem.*

##### Answer:

To train the model, I started LeNet architecture because of simplicity of implementation 
and because it performs well on recognition task with tens of classes (such as character recognition). 
After a few runs with this architecture I noted that the model tended to overfit to the original training set, 
in fact the learning curve showed that the training error converged to 99% while the validation error wasn`t giving 
a satisfactory performance. For this reasons, I tested two regularization techniques to improve the results:

* Data augmentation
* Dropout

I started trying with an high dropout rate 55% and this seemed to slow down overfitting: the model was slower to 
train but also achieved a slightly higher accuracy in the end. 
However, only When I added the augmented dataset I started seeing strong increased performance as the model 
was now able to learn within a few epochs but at the same time to generalize well on the validation set.

A dropout rate of 30% and a learning rate of 1e-4 
was selected after a few trial and errors. Training the model overall takes around 2 hour.

Training curves can be seen below, at the end of the curves both training and validation error converge around 
a hundred epochs.

![alt text][image5] 


My final model results were:
* training set accuracy of 97%
* validation set accuracy of 95%
* test set accuracy of 94%

### Test a Model on New Images

#### Question 1:

*Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.*

Here are five German traffic signs that I found on the web:

![alt text][image6]

All these images maybe challenging to classify because:

* They include much more background then the training images
* The background is very different from the one in the training images
* Contains image artifacts such as jpeg compression problems and copyright trademarks

Since these images are not in the right shape accepted by the classifier they were downsampled and smoothed before 
applying the `trasnform_img` function
 
 
Here are the results of the prediction:

Top 5 Labels for image `Double curve`:
 - `Road work` with prob = 0.96 
 - `Dangerous curve to the right` with prob = 0.01 
 - `Right-of-way at the next intersection` with prob = 0.01 
 - `Traffic signals` with prob = 0.01 
 - `Double curve` with prob = 0.00 
Top 5 Labels for image `Children crossing`:
 - `Speed limit (60km/h)` with prob = 0.99 
 - `Speed limit (50km/h)` with prob = 0.01 
 - `Speed limit (80km/h)` with prob = 0.00 
 - `Speed limit (30km/h)` with prob = 0.00 
 - `Speed limit (120km/h)` with prob = 0.00 
Top 5 Labels for image `Speed limit (50km/h)`:
 - `Speed limit (50km/h)` with prob = 0.97 
 - `Speed limit (80km/h)` with prob = 0.02 
 - `Speed limit (30km/h)` with prob = 0.01 
 - `Speed limit (60km/h)` with prob = 0.00 
 - `Speed limit (100km/h)` with prob = 0.00 
Top 5 Labels for image `Go straight or left`:
 - `Turn left ahead` with prob = 0.81 
 - `Priority road` with prob = 0.17 
 - `Keep right` with prob = 0.01 
 - `Right-of-way at the next intersection` with prob = 0.01 
 - `Ahead only` with prob = 0.00 
Top 5 Labels for image `Speed limit (80km/h)`:
 - `General caution` with prob = 0.77 
 - `Right-of-way at the next intersection` with prob = 0.21 
 - `Beware of ice/snow` with prob = 0.02 
 - `Priority road` with prob = 0.00 
 - `No entry` with prob = 0.00 


​
The model was able to correctly guess almost all of the traffic signs, which gives a good accuracy rate
This is very different from the accuracy on the test set but is also comprehensible given the different conditions in
which these images were take.

The model was realtively sure for the prediction for the first 3 images 
