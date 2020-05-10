function [A, B, C, cost_value] = PREMA_algorithm(Yt, Yc, U, V, W, lambda, max_iter, A, B, C)
%
% This function is an implementation of PREMA, Algorithm 1 in the paper titled: 
% "PREMA: Principled Tensor Data Recovery from Multiple Aggregated Views"
% It outputs the factors A, B, and C that are used to reconstruct the 
% disaggregated tensor X.  
%
% The inputs are:
%       Yt: is the temporally aggregated tensor, that is aggregated in the 
%           third mode by W, where W is an aggregation matrix.  
%       Yc: is the contemporaneously aggregated tensor, that is aggregated 
%           in the first mode by matrix U (and 2nd mode by matrix V), where
%           U and V are aggregation matrices.
%           Both Yt and Yc have NaN at the entries of missing data. 
%       U, V, and W: aggregation matrices as described above.
%
%       lambda: is the regularization parameter that controls the weight 
%               between the two aggregated tensors in the coupled 
%               factorization (set to 1 in our experiments).
%       max_iter: is the maximum number of iterations.
%       A, b, and C: are the intial values of the factors (the output of
%                    Initialization_of_PREMA.m function).
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
% Faisal Almutairi (almut012@umn.edu), Jan 2020




[I,J,Kw] = size(Yt);
[Iu,Jv,K] = size(Yc);

% inialize for UA, VB, and WC
A_tilde = U*A;
B_tilde = V*B;
C_tilde = W*C;

% get masks of the observed entries, then set missing entries to zero
maskYt = ~isnan(Yt);
maskYc = ~isnan(Yc);
Yt(~maskYt) = 0;
Yc(~maskYc) = 0;


% unfolding (metricization) over the 1st, 2nd, and 3rd modes of the two 
% tensors Yt and Yc, and their masks maskYt and maskYc    
Yt1 = reshape(Yt,[I,J*Kw])';
Yt2 = reshape(permute(Yt,[2 1 3]),[J,Kw*I])';
Yt3 = reshape(Yt,[I*J,Kw]);

Yc1 = reshape(Yc,[Iu,Jv*K])';
Yc2 = reshape(permute(Yc,[2 1 3]),[Jv,K*Iu])';
Yc3 = reshape(Yc,[Iu*Jv,K]);

maskYt1 = reshape(maskYt,[I,J*Kw])';
maskYt2 = reshape(permute(maskYt,[2 1 3]),[J,Kw*I])';
maskYt3 = reshape(maskYt,[I*J,Kw]);

maskYc1 = reshape(maskYc,[Iu,Jv*K])';
maskYc2 = reshape(permute(maskYc,[2 1 3]),[Jv,K*Iu])';
maskYc3 = reshape(maskYc,[Iu*Jv,K]);

cost_value = zeros(1,max_iter);
cost_value(1) = inf;
%% alternating between updating A, B, and C
for iter = 2:max_iter
%     disp(iter)
    %% update factor A
    kp_WCB = kr(C_tilde,B); % Khatri-Rao product function in the TensorLab package
    kp_CVB = kr(C,B_tilde);
    Res1 = (kp_WCB*A' - Yt1).*maskYt1;   
    Res2 = (kp_CVB*A_tilde' - Yc1).*maskYc1;  
    Res1_vec = -Res1(:); 
    Res2_vec = -Res2(:);    
    gradient_A = 2*Res1'*kp_WCB + 2*lambda*(Res2*U)'*kp_CVB;   
    D1 = (kp_WCB*gradient_A').*maskYt1; 
    D2 = (kp_CVB*(U*gradient_A)').*maskYc1;
    D1_vec = D1(:);
    D2_vec = D2(:);
    num = -Res1_vec'*D1_vec - lambda*Res2_vec'*D2_vec;
    denum = D1_vec'*D1_vec + lambda*(D2_vec'*D2_vec);
    alpha = max(0,num/denum);
    A = A - alpha*gradient_A;     
    A_tilde = U*A;        
    %% update factor B
    kp_WCA = kr(C_tilde,A);
    kp_CUA = kr(C,A_tilde);
    Res1 = (kp_WCA*B' - Yt2).*maskYt2; 
    Res2 = (kp_CUA*B_tilde' - Yc2).*maskYc2; 
    gradient_B = 2*Res1'*kp_WCA + 2*lambda*(Res2*V)'*kp_CUA;   
    Res1_vec = -Res1(:); 
    Res2_vec = -Res2(:);
    D1 = (kp_WCA*gradient_B').*maskYt2; 
    D2 = (kp_CUA*(V*gradient_B)').*maskYc2;
    D1_vec = D1(:);
    D2_vec = D2(:);
    num = -Res1_vec'*D1_vec - lambda*Res2_vec'*D2_vec;
    denum = D1_vec'*D1_vec + lambda*(D2_vec'*D2_vec);
    beta = max(0,num/denum);
    B = B - beta*gradient_B;
    B_tilde = V*B;
    %% update factor C
    kp_BA = kr(B,A);
    kp_VBUA = kr(B_tilde,A_tilde);
    Res1 = (kp_BA*C_tilde' - Yt3).*maskYt3;
    Res2 = (kp_VBUA*C' - Yc3).*maskYc3;
    gradient_C = 2*(Res1*W)'*kp_BA + 2*lambda*Res2'*kp_VBUA;    
    Res1_vec = -Res1(:); 
    Res2_vec = -Res2(:);
    D1 = (kp_BA*(W*gradient_C)').*maskYt3; 
    D2 = (kp_VBUA*gradient_C').*maskYc3;
    D1_vec = D1(:);
    D2_vec = D2(:);
    num = -Res1_vec'*D1_vec - lambda*Res2_vec'*D2_vec;
    denum = D1_vec'*D1_vec + lambda*(D2_vec'*D2_vec);
    gamma = max(0,num/denum);
    C = C - gamma*gradient_C;
    C_tilde = W*C;
     
    % compute the cost value
    f = norm(Yt3 - kp_BA*C_tilde','fro')^2;
    g = norm(Yc3 - kp_VBUA*C','fro')^2;
    cost_value(iter) = f + lambda*g;
    
    % stop when the cost value converges
    if abs(cost_value(iter)-cost_value(iter-1))/cost_value(iter) < eps
        break
    end

end

end