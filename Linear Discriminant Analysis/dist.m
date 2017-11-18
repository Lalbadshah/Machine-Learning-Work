function d = dist(a,b)
% E = dist(A,B)
%
%    A - (SxM) matrix 
%    B - (SxN) matrix
%    E - (MxN) Euclidean distances between vectors in A and B

A=sum(a.*a,1); 
B=sum(b.*b,1); 
AB=a'*b; 
d = sqrt(abs(repmat(A',[1 size(B,2)]) + repmat(B,[size(A,2) 1]) - 2*AB));