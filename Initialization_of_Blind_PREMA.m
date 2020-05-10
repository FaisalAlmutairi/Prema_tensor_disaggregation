function [A, A_tilde, B, C, C_tilde] = Initialization_of_Blind_PREMA(Yt, Yc, Rank, max_iter)  
% 
% This function is an implementation of the intialization step in 
% Algorithm 2 (B-PREMA) in the paper titled: 
% "PREMA: Principled Tensor Data Recovery from Multiple Aggregated Views".
% It outputs the factors A, B, and C that are used to initialize PREMA 
% algorithm.  
%
% The inputs are:
%       Yt: is the temporally aggregated tensor, that is aggregated in the 
%           third mode by an UNKNOWN aggregation matrix.  
%       Yc: is the contemporaneously aggregated tensor, that is aggregated 
%           in the first and 2nd modes by UNKNOWN aggregation matrices.
%           Both Yt and Yc have NaN at the entries of missing data. 
%       Rank: is the tensor rank.
%       max_iter: is the maximum number of iterations in the CPD step
%
%
% To run this code, you need to download TensorLab package (https://www.tensorlab.net) 
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
%
% Faisal Almutairi (almut012@umn.edu), Jan 2020


[I,J,Kw]=size(Yt);
[Iu,Jv,K]=size(Yc);


% set missing entries to zeros
Yt(isnan(Yt)) = 0;
Yc(isnan(Yc)) = 0;

 
%% initialization of the 5 variables (in eq. 24 in the paper)
% a) CPD of the contemporaneous aggregate Yc to get A_tilde, B, and C
[Fac,~] = cpd(Yc, Rank,'MaxIter',max_iter); % CPD function in the TensorLab package
A_tilde = Fac{1};
B = Fac{2};
C = Fac{3};

% b) estimate the temporal aggregation by summing every Nw=K/Kw consecutive
% time-ticks, to get C_tilte
W = zeros(Kw,K);
Nw = ceil(K/Kw);
for k = 1:Kw
    idx = (Nw*(k - 1) + 1):(Nw*k);
    W(k,idx) = ones(1,Nw);
end
[~, KK] = size(W);
W = W(:,(KK-K)+1:end); % crop if size exceeds the size of the third dimension in Yt
C_tilde = W*C; 

% c) solve the linear system using the temporal aggregat Yt to get A  
YY = reshape(Yt,[I,J*Kw])';
XX = kr(C_tilde,B);
A = (XX\YY)';


end