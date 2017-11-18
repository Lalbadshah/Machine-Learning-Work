#Linear Gaussian Classifier  - for handwritten digit classification

Files in io are not mine and are simply used to load the MNIST data set and its respective labels that are stored in the data folder

Run the LGA_main.m - driver script first It will take about about 6 mins to run

It calls the LGC function for each set number of eigen vectors

The LGC function takes the number of eigen vectors to be used as input and outputs the Accuracy using shared covariance and Acuracy using individual covariances

Observation -
For the case of shared covariance there is not much change with respect to number of eigen vectors used
For the case of Individual Cavariances used as the number of eigen vectors used increase the accuracy of the classification decreased noticable

These changes were observed from the graph plotted by the LGA_main.m script

Here I work with the MNIST database. MNIST is a standard and large database of handwritten digits. MNIST dataset has been widely used as a benchmark for testing classification algorithms in handwritten digit recognition systems[http://yann.lecun.com/exdb/mnist/].  MNIST is an abbreviation for Mixed National Institute of Standards and
Technology database. This database is created by “re-mixing” the original samples of NIST’s database. The database has two parts: training samples that was taken from American Census Bureau employees and the test samples that was taken from American high school students.
