# Real Time 2D Object Recognition
 

# Brief description
The purpose of this project was to explore the field of real-time 2D object recognition by developing a system that can identify specific objects placed on a white surface, regardless of their size, orientation, or position, using a camera positioned overhead. The project involved several steps, including converting the image to a binary format through thresholding, removing unwanted noise, and identifying the object's region of interest. We then computed feature vectors of the objects and used them to train our system. Finally, we developed a recognition system that uses two different classifiers to identify objects in single images or video sequences in real-time. The system is capable of outputting an image with identified objects for single images and real-time video sequences.

***
# Threshold the input video

I have implemented a thresholding algorithm from scratch. This function applies thresholds on the input image and returns the result. The function takes in the RGB image, converts it to grayscale and then creates a binary image using threshold. The background will be black after the threshold application. Thresholded Images:

![image](https://github.com/Atharva-Pandkar/Real-Time-2D-Object-Recognition/assets/62322017/3dce2045-62e7-4c41-b145-32863cbc4117)
***
# Clean up the binary image

After obtaining a binary image through thresholding, it is common to observe holes and gaps in the object area and salt and pepper noise in the background. To address this, morphological operators such as erosion and dilation are often applied. Additionally, the morphology function, which utilizes erosion and dilation as basic operations, can be used to perform more advanced morphological transformations. The output of morphology is typically an outline of the object, achieved by taking the difference between the dilation and erosion of the image.
![image](https://github.com/Atharva-Pandkar/Real-Time-2D-Object-Recognition/assets/62322017/c955f6d4-9521-41ea-aab4-767df759aab2)

***
# Segment the image into regions

Although the binary images obtained from previous processing steps have been cleaned up, there may still be areas in the foreground that are not relevant to our analysis. To isolate and label the regions of interest, image segmentation is necessary. The connected components with the stats function were used on the cleaned binary image to segment it into regions, and the resulting statistics were used to create bounding boxes that could be displayed on the image
![image](https://github.com/Atharva-Pandkar/Real-Time-2D-Object-Recognition/assets/62322017/62713822-95ad-4867-a6c4-0bffaa1f06d4)

***
# Compute features for each major region

To extract useful information from the segmented image, the findContours function provided by OpenCV was utilized to obtain the contours of the objects. Then, the minAreaRect function was applied to compute the rectangle area, extract its points, and draw an oriented bounding box using these 4 points. Subsequently, important features that are translation, rotation, scale, and reflection invariant were computed and stored in a database. These features include the percentage of the bounding box filled, the aspect ratio, and the Hu moments.
![image](https://github.com/Atharva-Pandkar/Real-Time-2D-Object-Recognition/assets/62322017/7ab8328d-197e-4122-a2f4-d6e247928b1f)

***
# Collect training data

To collect the features of all the objects in an image and store them in a database, a separate file named training.cpp was created. The user inputs the path of the images, and the system computes all the features of that image and stores them in the features_database.csv file. To accomplish this, csv_util header files provided by the professor for Project 2 were used.
For real-time training, the skeleton code provided for capturing video in Project 1 was utilized. The extracted features are then stored in the features_database.csv file, which now serves as the training data for object classification in images.
![image](https://github.com/Atharva-Pandkar/Real-Time-2D-Object-Recognition/assets/62322017/52622a4e-9095-499b-9a8c-2c3e57dbae69)

***
# Classify new images

To classify new objects, the distance between their feature vectors and all feature vectors stored in the features_database.csv file is calculated using a scaled Euclidean distance metric. The distances are then sorted, and the object is assigned a label based on the feature vector with the least distance
![image](https://github.com/Atharva-Pandkar/Real-Time-2D-Object-Recognition/assets/62322017/c7236f26-662a-46ff-8fbe-c2d3be6436f9)

***
# Use a different classification

To achieve more accurate results, a KNN classifier was implemented. This function calculates the distances from all instances of each object, selects the top k distances, adds them to obtain the final distance from that object class, and then assigns the object the label of the object class with the least distance.
