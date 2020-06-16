Digit-Recognizer

We develop a   means   to classify   a   given   image   of   a handwritten number into one of ten classes.  These ten classes being single digit integers 0 to 9. Using Machine Learning (ML) concepts  we  can  approach  this  situation  in  various  ways.  By using:   Neural   Networks(NN),   Support   Vector   Machines (SVM), and K-nearest neighbors (KNN). To begin, we look at the MNIST dataset, which contains 60,000 handwritten images of theintegers from 0 to 9. MNIST stands for modified, natural institute of standards and technology dataset. 

MNIST Dataset(https://en.wikipedia.org/wiki/MNIST_database)

The train and test data are obtained from Kaggle(https://www.kaggle.com/oddrationale/mnist-in-csv) and it contains 784 pixel values in columns and label for each digit.

MNIST handwriting digits classification using different algorithms:

1. Support Vector Machine(SVM) polynomial kernel

Using only 12000 training samples (out of 60000), 96% score is obtained and on the test data the score is 93%. Best parameters for SVM are determined using GridSearchCV on the training data-set and the obtained parameters are (C = 0.001, $\gamma$ = 10). These parameters are used for test data. Polynomial kernel takes the least time and gives the best result following my codes.

2. K-nearest-neighbour(K-NN) algorithm

Making the image of digits to an N-dimension digital array and using the KNeighborsClassifier to fit the relationship between pixels(N dimension caracteristics) and labels(number: from 0 to 9).

3. Convolutional Neural Network(CNN) classifier

Used a convolutional neural network with 2 convolution layers followed by two fully connected layers.


