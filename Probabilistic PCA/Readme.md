#Probabilistic Principal Component Analysis - Expectation Maximization version of PCA

Run the PPCA_main.m script first It will take less than 3 mins to run (mine took 2mins and 50 secs)

It calls the PPCA function for each set number of eigen vectors [50 to 200] in step of 50 and
prints the accuracy for each number of eigen vectors used based on a Linear Gaussian Classifier - that I wrote previously

Files - loadMNISTImages.m and loadMNISTLabels.m are not mine and are only used to load the MNIST data

Here I work with the MNIST database. MNIST is a standard and large database of handwritten digits. MNIST dataset has been widely used as a benchmark for testing classification algorithms in handwritten digit recognition systems[http://yann.lecun.com/exdb/mnist/].  MNIST is an abbreviation for Mixed National Institute of Standards and
Technology database. This database is created by “re-mixing” the original samples of NIST’s database. The database has two parts: training samples that was taken from American Census Bureau employees and the test samples that was taken from American high school students.
