# TEETH-SEGMENTATION

The project was based on an X-ray teeth images dataset, where I have the input images (X-ray) and the segmentation masks (outputs). Each tooth is represented with a different color in grayscale.
All the project was made with TensorFlow Keras.

![image](https://github.com/guilhermegobbo/TEETH-INSTANCE-SEGMENTATION/assets/136920721/a027b693-7b5b-42a5-94e2-3024fc8ba435)

### U-NET MODEL

The U-Net model is mainly used for image segmentation tasks because of it's good performance on small and large datasets. If you have a very small dataset, you can also fine-tune your model by retraining a model that was originally trained on a larger dataset for a similar task. This is beneficial because you can freeze some layers during the retraining process, allowing you to avoid overfitting and reduce computational costs without compromising accuracy.

It's architechture:
![image](https://github.com/guilhermegobbo/TEETH-INSTANCE-SEGMENTATION/assets/136920721/b1174758-cb22-4b8f-88b7-eb6ff9f0767c)

### TRAINING PROCCESS

For the training, I applied augmentation by rotating and slightly zooming the images to enhance accuracy on the test dataset predictions. I used a batch size of 4, ran 10 epochs, set the learning rate to 0.001, and specified a height/width of the output image as 512 pixels with 33 filters (considering that a person has 32 teeth, and accounting for the 0 pixel values, the total is 33 filters).

![image](https://github.com/guilhermegobbo/TEETH-INSTANCE-SEGMENTATION/assets/136920721/295a5390-9968-4ab5-a13e-614ddaf03fc7)

(The epoch didn't take 29,000 seconds hahaha. It's because when the notebook sleeps, the process of fitting pauses, returning just when I go back.)

### PREDICTIONS

Now, the most interesting part: the predictions. An image speaks for itself, so I'll just let the image below.

![image](https://github.com/guilhermegobbo/TEETH-INSTANCE-SEGMENTATION/assets/136920721/41ca1431-7da1-429a-91a0-15f7d9eb5230)


If you liked it, please give a star to the project to let me know, or feel free to contact me, I'll be so happy!
Thank you!
