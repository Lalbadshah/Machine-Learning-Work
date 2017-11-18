function [ acc] = accuracy( Ind )
k=1;
ac=0;
%--------Calculating accuracy By seeing if the index with the minimum
%distance matches with the index interval for each class-------------
for i=1:9:352
    if(Ind(k)<=i+8 && Ind(k)>=i)
        ac=ac+1;
    end
    k=k+1;
end
acc = (ac/40)*100;
end

