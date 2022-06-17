# Written Response Questions

#### 1. According to our research, a machine learning algorithm has an accuracy of 80%. However, when we applied the algorithm to practical use, the accuracy was below the expected accuracy. Describe two or more causes to this.

In practical use cases, data comes from various sources and conditions. Data used for test and validation usually are very similar to training data. Actually in most of the cases all the training, test and validation data come from the same source. So, there are chances that the performance depicted by the evaluation metrics is not indicative of the practical cases where the sources of data fed into the model might be different.

Raw data from different sources might require different preprocessing before being fed into the model. Also, the data might change with time. Having a rigid ML pipeline for all data sources can be another reason for the reduction of accuracy. 

Also, accuracy may not be the best metric to represent the model. For imbalanced data, accuracy is deceiving. The model should be evaluated using different metrics to make sure that the model is underperforming. 

#### 2.  Please suggest and describe two industry applicable Deep Learning projects you would carry out using a dataset of aerial images.

One industry application project involving aerial images I would like to work on can be agriculture. Agriculture applications require accurate land monitoring for support and control of growth of plants which can be done with drone images. Aerial images can also be used to analyze height, plant shape and many other phenotypes in depth. This automatic and easy crop analysis across huge and diverse geographic areas can save farmers from a lot of burden.  
	
Another application project of deep learning in aerial images can be object and person tracking in sports. With further development of onboard camera and processing in the drone. The drone can follow and track peoples and objects real time in unique patterns giving new and unique viewing experience to the audience. Also the tracking can be used for analysis of a players game.  


#### 3. You have a large data set consisting of high-resolution aerial orthophotos. Your objective is to create an API that detects small objects within an orthophoto (e.g. Trees, Cars, People, etc). Please explain how you would create a Deep Learning Pipeline by elaborating on how you would approach the following steps. (No more than 300 words in total)

- Data Preprocessing/Labeling:
	Since we have a huge dataset, we can directly move on to labeling the data without augmenting it. For labeling the data we can use different data labeling softwares like 'roboflow' or 'labelImg'. Labeling a huge dataset is usually a manual and tedious task. According to our algorithm of choice, the annotation of the labeled data can be stored in different formats. Every image will have seperate annotation file. We should then divide the data into train, test and validation set. Finally, we should place the images and annotation files appropriately as per the requirement of the model.
	
- Model Selection:
	Since we are to detect small objects within the photo, the performance of the model should be good. The model should have high mAP. Though the inference time is of secondary importance in our case, a model with smaller inference time on top of high mAP would be ideal. Object detection models like YOLOR(mAP 56.1), YOLOv4(mAP 55.4), EffificentDet(mAP 55.1) are currently the models with highest mAP (in MS-COCO dataset). In terms of speed as well, YOLOv4 is a top performer with a fairly good inference time of 12s. So, in our case, YOLOv4 will be a good choice.
	
- Model Training:
	For training the model, we will use transfer learning. For that we will need to clone the original repository and download pretrained weight file trained on MS-COCO dataset. Then we will have to change the configurations of the model according to the number of classes the model will be trained on. Then we must build darknet and train the model with custom configurations. The training can be stopped when the avergte loss is very small and doesn't show any significant change for a while.

- Model Optimization/Hyperparameter Tuning:
	The evaluation of our trained model is given by mAP. The higher the mAP the better our model is at detecting objects. We can further improve this performance with hyperparameter tuning. In the configuration file, we can change the hyperparameters like batch size, learning rate, number of iterations, number of layers and filters so as to improve the performance of the model.

- Model Hosting/Deployment/Management:
	Our model can be deployed directly on edge devices or be hosted on web or mobile apps. For making the model accessible to client, APIs capable of processing both videos and images should be created and hosted in website. 
