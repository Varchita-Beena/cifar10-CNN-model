# cifar10-CNN-model
## Cifar10 dataset <br>
Cifar10 dataset contains 6000 images of 10 object categories. Dataset is available https://www.cs.toronto.edu/~kriz/cifar.html. <br>

## Approach <br>
In total two convolutional layers Conv2D-48 and Conv2D-96 each followed by approritate activation function given in table maxpooling of kernel 2 size and dropout of 0.5. <br>
The ones having batch normalization are added after activation function. In last three rows in table one more Conv32D-192, maxpool of kernel 2 and dropout of 0.5 is used.<br>
In without batch normalization and mometum and adaptive learning rates optimizer adam is used and categorical crossentropy is used and in without batch normalization and mometum and adaptive learning rates optimizdder SGD, learning rate=0.01, decay=1e-6, momentum=0.9, nesterov and loss mean squared error isused.<br>

## Machine <br>
Training is done on 50000 samples and testing on 10000 samples. This is ran on processor Intel Core (TM) i5-4210U CPU @ 1.70GHz with installed RAM 4.00GB.<br>

## Analysis <br>
From table:- <br>
(1) We can see that ReLU as activation function is giving best accuracy w.r.t other activation functions. <br>
(2) As number of epochs increase the accuracy will increase. <br>
(3) Time taken is also less for RelU in without Batch normalization, momentum and adaptive learning rates.  <br>
(4) For both sigmoid and tanh activation functions accuracy increases for Batch normalization, momentum and adaptive learning rates and time also increases than simple sigmoid and tanh but that is worth taking.<br>

![Results](anaysis.png)
