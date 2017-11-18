function [] = showImg(X,C,I,K)
Image=zeros(size(X,1)*size(X,2),size(X,3));
for k=1:K
    for l=1:size(X,1)*size(X,2)
        if(I(l)==k)    
            Image(l,:)=C(k,:);
        end
    end
end
figure;
imshow(uint8(reshape(Image,size(X,1),size(X,2),size(X,3))));
end