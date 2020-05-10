function [A, B, C] = Initialization_of_PREMA(Yt, Yc, U, V, W, Rank, max_iter)  
% 
% This function is an implementation of the algorithm in APPENDIX C in the 
% paper titled: 
% "PREMA: Principled Tensor Data Recovery from Multiple Aggregated Views".
% It outputs the factors A, B, and C that are used to initialize PREMA 
% algorithm.  
%
% The inputs are:
%       Yt: is the temporally aggregated tensor, that is aggregated in the 
%           third mode by W, where W is an aggregation matrix.  
%       Yc: is the contemporaneously aggregated tensor, that is aggregated 
%           in the first mode by matrix U, and in the 2nd mode by matrix V, 
%           where U and V are aggregation matrices.
%           Both Yt and Yc have NaN at the entries of missing data. 
%       U, V, and W: aggregation matrices as described above.
%       Rank: is the tensor rank.
%       max_iter: the maximum number of iterations in the CPD step.
%
%
% To run this code, you need to download TensorLab package (https://www.tensorlab.net) 
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
%
% Faisal Almutairi (almut012@umn.edu), Jan 2020



% set missing entries to zeros
[I,J,Kw]=size(Yt);
[Iu,Jv,K]=size(Yc);
Yt(isnan(Yt)) = 0;
Yc(isnan(Yc)) = 0;

if (J == Jv) && (K >= I)
    % case 1: if Yc is only aggregated over the first mode, and K >= I 
    % a) CPD of the contemporaneous aggregate Yc to get B and C
    [Fac,~] = cpd(Yc, Rank, 'MaxIter', max_iter); % CPD function in the TensorLab package
    A_tilde = Fac{1};
    B = Fac{2};
    C = Fac{3};

    % b) solve the linear system using the temporal aggregat Yt to get A  
    C_tilde = W*C;
    B_tilde = V*B;
    XX = kr(C_tilde,B);
    YY = reshape(Yt,[I,J*Kw])';
    A = (XX\YY)';
else
    
    % case 2: if Yc is aggregated over the first AND second modes, or K < I 
    % a) CPD of the temporal aggregate Yt to get A and B
    [Fac, ~] = cpd(Yt, Rank, 'MaxIter', max_iter); % CPD function in the TensorLab package
    A = Fac{1};
    B = Fac{2};
    C_tilde = Fac{3};

    % b) solve the linear system using the temporal aggregat Yc to get C  
    A_tilde = U*A;
    B_tilde = V*B;
    XX = kr(B_tilde,A_tilde);
    YY = reshape(Yc,[Iu*Jv,K]);
    C = (XX\YY)';
end