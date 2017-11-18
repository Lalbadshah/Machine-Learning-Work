X= imread('elephant.jpg');
[C,I] = myKmeans(X,2,100);
showImg(X,C,I,2);
[C,I] = myKmeans(X,5,100);
showImg(X,C,I,5);
[C,I] = myKmeans(X,10,100);
showImg(X,C,I,10);