# SHRED-Model-Analysis

## Abstract

In this assignment, we will be taking a look at the SHRED model. The SHRED model takes in data to figure out the sea-surface temperature of the oceans. The assignment is to change the values of the number of sensors, the time lag variable, and add noise to the data. We will analyze the performance of the model to the grounded truth.

## Introduction

The SHRED model takes in data from the NOAA and uses machine learning to figure out the sea-surface temperature. The SHRED model code is from Github and created by Jan P. Williams, Olivia Zahn, J. Nathan Kutz, and the username shervinsahba for the graphing portion of the code. We will be changing the values of the number of sensors, the time lag variable, and adding noise to the data. We will analyze the performance of each individual variable and compare it to the grounded truth. 

## Theoretical Background

The idea is to take in sensor data from random points of the world’s oceans. Afterward, the data will provide the sensors data and the measured X values from the sensors. We want to have a function model where we input the sensor data and predict the x-value data. The difference between the predicted X values and the actual X values would be minimized based on the function. The model uses the LSTM neural network architecture. The overall data and model contain the state space, measurements from the sensor, and mapping. We want to optimize the model so that we can predict the sea-surface temperature with just a few sensors randomly dotted around the map. 

## Algorithm, Implementation, and Development

The main values we want to change are the lag time value, the number of sensors, and adding a Gaussian noise for the data. The model requires us to run a certain epoch. However, my computer's CPU is slow and does not support CUDA. Therefore, I had to create the code from Shervinsahba and import it to Google Colab. Then, I set the runtime type to GPU and the code is running at a reasonable speed. 

The majority of the code is the same as the code that was given to us in class. However, this requires us to add noise to the model. Therefore, I added the Gaussian function to add noise to the load_X data. 

Next, the data is divided into training, validation, and testing data. We use the MinMaxScalar package to preprocess the data for input and the output of the training, validation, and testing data. Then, we input the training and validation data into the model and train the model for testing. The final output should be the mean squared error by comparing it to the grounded truth. The graph shows the model’s version of the sea-surface temperature and the grounded truth helps with the comparison. 

## Results

For the base case scenario, we have the lag time set for 52 weeks in a year, the number of sensors to 3, and no noise to the data. The resulting mean squared error is **0.020089379**
 and the graph is shown in Figure 1. 
 
 ![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/BaseCase.png)
 
 Figure 1: The base case of the SHRED model. Lag time is 52, number of sensor is 3, and no noise.

Now, we will have different values of the lag time to see the performance of the model. First, we change the lag time to 4 weeks. The resulting mean squared error is **0.039661653** and the graph is shown in Figure 2. We can see that reducing the lag time increases the mean squared error. Now, changing the value to 104, we get a resulting mean square error of **0.01972125**. Figure 3 shows the resulting graph of the lag time 104. 

![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/4W%20Case.png)
 
 Figure 2: The 4 week case of the SHRED model. Lag time is 4, number of sensor is 3, and no noise.
 
 ![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/104W%20Case.png)
 
 Figure 3: The 104 week case of the SHRED model. Lag time is 104, number of sensor is 3, and no noise.
 

Next, let's add different levels of noise. For the Gaussian noise function, we will be changing the standard deviation of the random code. For a standard deviation of 10, the mean squared error was **0.49616945**. Figure 4 shows a drastic change to the recon graph. The sea-surface temperature is shown to be noisy compared to the ground truth. If we were to increase the standard deviation of the noise levels, the mean square error would increase as well. For example, we raised the standard deviation to 15 and the mean square error is **0.64647007**. Figure 5 shows a more noisy version of the ground truth. 

![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/10N%20Case.png)
 
Figure 4: The standard deviation of 10 case of the SHRED model. Lag time is 52, number of sensor is 3, and noise level with standard deviation of 10.
 
![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/15N%20Case.png)
 
Figure 5: The standard deviation of 15 case of the SHRED model. Lag time is 52, number of sensor is 3, and noise level with standard deviation of 15.

Finally, we want to see what happens when we change the number of sensors. First, we change the lag time back to 52 weeks. Then, we reduce the number of sensors to one. The resulting mean square error is **0.020238835**. The mean square error is a little larger than the base case but Figure 6 shows similar results. The next number of sensors is something larger than the base case, therefore, I chose 9 sensors to see the performance of the model. The resulting mean squared error is **0.019495467**. The resulting increase and decrease in the number of sensors make sense as the more data you have the better the model can predict the grounded truth. Figure 7 shows the performance of the 9 sensors. 

![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/1S%20Case.png)
 
 Figure 6: The 1 sensor case of the SHRED model. Lag time is 52, number of sensor is 1, and no noise.
 
![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/9S%20Case.png)
 
 Figure 7: The 9 sensor case of the SHRED model. Lag time is 52, number of sensor is 9, and no noise.
 
 For a better look at the results, I made a loop to create a graph when changing the values. For Figure 8, 9, and 10, we can see the neural network can still function even though the number of sensors and lag time changes. However, the noise levels increases the mean square error as we increase the standard deviation. 
 
 ![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/Error%20for%20Lag.png)
 
 Figure 8: The mean square error stays roughly the same as we increase the lag time. 
 
 ![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/Error%20for%20Sensor.png)
 
 Figure 9: The mean square error stays roughly the same as we increase the number of sensors
 
 ![](https://github.com/SamQLuong/SHRED-Model-Analysis/blob/main/Error%20for%20Noise.png)
 
 Figure 10: The mean square error increases as we increase the standard deviation of the noise levels.

## Conclusion

The purpose of this assignment is to see what happens when we change the lag time, number of sensors, and noise levels. By comparing the base case to the other cases, we can see the difference in mean square error. The power of the neural network can still have good enough results even if we decrease the time lag. The lag time shows that even if we decrease the value to 4 weeks, the results would be roughly **0.039661653**. However, the larger the lag time the better the results of mean squared error. The resulting mean squared error for 104 weeks was **0.01972125**.

The number of sensors had little effect on the mean squared error when we decreased the number of sensors. However, the more sensors we have, the better the results would be. The resulting mean squared error with a larger number of sensors was **0.019495467**. 

The only great effect on the model was noise. For example, if we were to have a standard deviation of 10 to the noise level, then the resulting mean squared error will be **0.49616945**. The mean squared error would increase as we increased the standard deviation of the noise level. 

Overall, the neural network is powerful enough to be as close as possible to the grounded truth even if we change the number of sensors or lag time. However, the model is weak when the data is hit with noise. 

All the code is from the two github accounts https://github.com/Jan-Williams/pyshred and https://github.com/shervinsahba/pyshred


