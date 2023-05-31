from django.shortcuts import render, redirect
from myapp.models import UploadedFile
from django.core.files.storage import default_storage
import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Multiply, GlobalAveragePooling2D, Reshape, Dense, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

model_path = 'C:/Users/nihal/Desktop/Academics/Main/MainProject/Main_Project/My_Project/CrowdCount/Crowd_models/model.h5'
model = tf.keras.models.load_model(model_path)
        # Define the input shape of the model
input_shape = (224, 224, 3)  # replace with the actual shape of your feature map

def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video_file')
        file_path = default_storage.save(video_file.name, video_file)
        video_path = default_storage.path(file_path)

        # if video_file:
        #     file_path = default_storage.save(video_file.name, video_file)
        #     # The file_path variable now contains the path where the video file is saved in the local storage
        #     # You can store this file_path in your model or perform further processing
            
        #     # Example: Saving the file path in the UploadedFile model
        #     uploaded_file = UploadedFile(file_path=file_path)
        #     uploaded_file.save()

        # Open the video file
        cap = cv2.VideoCapture(str(video_path))

        # Initialize the headcount
        headcount = 0

            # Read the next frame
        # Read the next frame
        ret, frame = cap.read()

        # Resize the frame to the input shape of the model
        resized = cv2.resize(frame, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize the pixel values
        normalized = resized.astype('float32') / 255.0

        # Expand the dimensions of the frame to match the input shape of the model
        tensor = np.expand_dims(normalized, axis=0)
            
            # Make a prediction using the model
        prediction = model.predict(tensor)[0]
            
            # Add the predicted headcount to the total headcount
        headcount += prediction
        cap.release()
        headcount = int(sum(headcount))
        print('Total headcount:', headcount)
    
        # if 'upload' in request.POST:  # Handle Upload Video button
        #     if video_file:
        #         file_path = default_storage.save(video_file.name, video_file)
        #         UploadedFile.objects.create(file_path=file_path)
        #         return render(request, 'upload.html', {'upload_success': True})
        # elif 'get_crowd_count' in request.POST:  # Handle Get Crowd Count button
        #     latest_file = UploadedFile.objects.latest('upload_time')
        #     video_path = default_storage.path(latest_file.file_path)


        #     total_headcount = calculate_crowd_count(model_path, video_path)
        default_storage.delete(file_path)
        return render(request, 'result.html', {'headcount': headcount})

    return render(request, 'upload.html')

# def calculate_crowd_count(model_path, video_path):
#     # Load the trained model
#     model = tf.keras.models.load_model(model_path)

#     # Define the input shape of the model
#     input_shape = (16, 224, 224, 3)  # replace with the actual shape of your feature map

#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Initialize the headcount
#     headcount = 0

#     # Loop over all frames in the video
#     while cap.isOpened():
#         # Read the next frame
#         ret, frame = cap.read()

#         # Check if the frame was read successfully
#         if not ret:
#             break

#         # Resize the frame to the input shape of the model
#         resized = cv2.resize(frame, (input_shape[1], input_shape[2]), interpolation=cv2.INTER_LINEAR)

#         # Normalize the pixel values
#         normalized = resized.astype('float32') / 255.0

#         # Add a batch dimension to the input tensor
#         tensor = np.expand_dims(normalized, axis=0)

#         # Make a prediction using the model
#         prediction = model.predict(tensor)[0]

#         # Add the predicted headcount to the total headcount
#         headcount += prediction

#     # Release the video file
#     cap.release()

#     # Calculate the total headcount
#     total_headcount = int(np.sum(headcount))
#     return total_headcount
