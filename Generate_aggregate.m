function [Yt, Yc] = Generate_aggregate(X, U, V, W)
% 
% This function generates aggregated tensors from the high-resolution 
% tensor X, In particular:
%       Yt: is the temporally aggregated tensor, that is aggregated in the 
%           third mode by W. 
%       Yc: is the contemporaneously aggregated tensor, that is aggregated 
%           in the first mode by matrix U AND 2nd mode by matrix V. 
%       U, V, and W: are aggregation matrices.
% X has NaN at the entries of missing data. For simplicity, if a missing 
% entry in X is to be summed with other entries that are available, we 
% assume that this sum is missing and will be NaN in Yt or Yc.          
%
%
% Note that the aggregation is done for each entry individually instead of
% using matrix multiplication to avoid errors caused by NaN in X.
%
%
% 
% 
% 
% Ref. 1: Almutairi, F.M., Kanatsoulis, C.I., and Sidiropoulos, N.D., 
% "PREMA: Principled Tensor Data Recovery from Multiple Aggregated Views." 
% arXiv preprint arXiv:1910.12001, 2019.
%
% Ref. 2: Almutairi F.M., Kanatsoulis C.I., Sidiropoulos N.D., "Tendi: Tensor 
% Disaggregation from Multiple Coarse Views," In Proc. of The Pacific-Asia 
% Conference on Knowledge Discovery and Data Mining (PAKDD), 2020.
%
%
% Faisal Almutairi (almut012@umn.edu), Jan 2020



[I, J, K] = size(X);
[Iu, ~] = size(U);
[Jv, ~] = size(V);
[Kw, ~] = size(W);

Yt = zeros(I, J, Kw);
Yc = zeros(Iu, Jv, K);


for k = 1:Kw
 idxk = find(W(k,:));
 for i = 1:I
    for j = 1:J
        Yt(i,j,k) = sum(X(i,j,idxk));
    end
 end
end           

for i = 1:Iu
 idxi = find(U(i,:));
 for j = 1:Jv
     idxj = find(V(j,:));
    for k = 1:K
        Yc(i,j,k) = sum(X(idxi,idxj,k));
    end
 end
end       


end