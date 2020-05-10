function [A, B, C, cost_value] = Blind_PREMA_algorithm(Yt, Yc, lambda, mu, max_iter, A, A_tilde, B, C, C_tilde)
%
% This function is an implementation of B-PREMA, Algorithm 2 in the paper titled: 
% "PREMA: Principled Tensor Data Recovery from Multiple Aggregated Views"
% It outputs the factors A, B, and C that are used to reconstruct the 
% disaggregated tensor X.  
%
% The inputs are:
%       Yt: is the temporally aggregated tensor, that is aggregated in the 
%           third mode by an UNKNOWN aggregation matrix.  
%       Yc: is the contemporaneously aggregated tensor, that is aggregated 
%           in the first and 2nd modes by UNKNOWN aggregation matrices.
%           Both Yt and Yc have NaN at the entries of missing data. 
%
%       lambda: is the regularization parameter that controls the weight 
%               between the two aggregated tensors in the coupled 
%               factorization (set to 1 in our experiments).
%       mu = the regularization parameter of the scaling ambiguity
%            regularizer in eq.24 in the paper (set to 100 in our experiments)
%       max_iter = he maximum number of iterations of B-PREMA algorithm.
%       A, A_tilde, B, C, C_tilde: intial values of the five variables (the 
%                         output of Initialization_of_Blind_PREMA.m function). 
%
%
%
% To run this code, you need to download TensorLab package (https://www.tensorlab.net) 
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
% 
% Faisal Almutairi (almut012@umn.edu), Jan 2020




[I,J,Kw]=size(Yt);
[Iu,Jv,K]=size(Yc);

% get masks of the observed entries, then set missing entries to zero
maskYt = ~isnan(Yt);
maskYc = ~isnan(Yc);
Yt(isnan(Yt)) = 0;
Yc(isnan(Yc)) = 0;

% unfolding (metricization) over the 1st, 2nd, and 3rd modes of the two 
% aggregated tensors (Yt and Yc), and their masks (maskYt and maskYc)  
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
%% alternating between updating the five factors
for iter = 2:max_iter
%     disp(iter)
    %% update factor A
    kp_WCB = kr(C_tilde,B); % Khatri-Rao product function in the TensorLab package
    Res1 = (kp_WCB*A' - Yt1).*maskYt1;         
    gradient_A = 2*Res1'*kp_WCB;  
    Res1_vec = -Res1(:);
    D = reshape((kp_WCB*gradient_A').*maskYt1,[I*J*Kw,1]);
    D_vec = D(:);
    num = (-Res1_vec'*D_vec);
    denum = (D_vec'*D_vec);
    alpha = max(0,num/denum);
    A = A - alpha*gradient_A;     
    
    %% update factor A_tilde
    kp_CVB = kr(C,B); 
    Res2 = (kp_CVB*A_tilde' - Yc1).*maskYc1;       
    gradient_A_tilde = 2*lambda*(Res2)'*kp_CVB;    
    Res2_vec = -Res2(:);
    D = reshape((kp_CVB*(gradient_A_tilde)').*maskYc1,[Iu*Jv*K,1]);
    D_vec = D(:);
    num = -lambda*Res2_vec'*D_vec;
    denum = lambda*(D_vec'*D_vec);
    rho = max(0,num/denum);
    A_tilde = A_tilde - rho*gradient_A_tilde;     
    
    %% update factor B
    kp_WCA = kr(C_tilde,A);
    kp_CUA = kr(C,A_tilde);
    Res1 = (kp_WCA*B' - Yt2).*maskYt2; 
    Res2 = (kp_CUA*B' - Yc2).*maskYc2; 
    gradient_B = 2*Res1'*kp_WCA + 2*lambda*(Res2)'*kp_CUA;   
    Res1_vec = -Res1(:); 
    Res2_vec = -Res2(:);
    D1 = reshape((kp_WCA*gradient_B').*maskYt2,[I*J*Kw,1]); 
    D2 = reshape((kp_CUA*(gradient_B)').*maskYc2,[Iu*Jv*K,1]);
    D1_vec = D1(:);
    D2_vec = D2(:);
    beta = max(0,((-Res1_vec'*D1_vec-lambda*Res2_vec'*D2_vec)/(D1_vec'*D1_vec+lambda*(D2_vec'*D2_vec))));
    B = B - beta*gradient_B;
     
    %% update factor C
    kp_VBUA = kr(B,A_tilde);
    Res2 = (kp_VBUA*C' - Yc3).*maskYc3;
    gradient_C = 2*lambda*Res2'*kp_VBUA + 2*mu*(ones(K,1)*ones(1,K)*C - ones(K,1)*ones(1,Kw)*C_tilde);    
    Res1_vec = (ones(1,Kw)*C_tilde - ones(1,K)*C)'; 
    Res2_vec = -Res2(:); 
    D1_vec = (ones(1,K)*gradient_C)';
    D2 = (kp_VBUA*gradient_C').*maskYc3;
    D2_vec = D2(:);
    num  = -mu*Res1_vec'*D1_vec-lambda*Res2_vec'*D2_vec;
    denum = mu*(D1_vec'*D1_vec)+lambda*(D2_vec'*D2_vec);
    gamma = max(0,num/denum);
    C = C - gamma*gradient_C;
    
    %% update factor C_tilde
    kp_BA = kr(B,A);
    Res1 = (kp_BA*C_tilde' - Yt3).*maskYt3;
    gradient_C_tilde = 2*(Res1)'*kp_BA + 2*mu*(ones(Kw,1)*ones(1,Kw)*C_tilde - ones(Kw,1)*ones(1,K)*C);   
    Res1_vec = -Res1(:); 
    Res2_vec = (ones(1,Kw)*C_tilde - ones(1,K)*C)';
    D1 = (kp_BA*(gradient_C_tilde)').*maskYt3; 
    D1_vec = D1(:);
    D2_vec = (ones(1,Kw)*gradient_C_tilde)';
    num = -Res1_vec'*D1_vec-mu*Res2_vec'*D2_vec;
    denum = D1_vec'*D1_vec+mu*(D2_vec'*D2_vec);
    sigma = max(0,num/denum);
    C_tilde = C_tilde - sigma*gradient_C_tilde;
    
         
    % compute the cost value
    f = norm(Yt3-kp_BA*C_tilde','fro')^2;
    g = norm(Yc3-kp_VBUA*C','fro')^2;
    r = norm(ones(1,K)*C - ones(1,Kw)*C_tilde)^2; 
    cost_value(iter)= f + lambda*g+ + mu*r;
    
    % stop when the cost value converges
    if     abs(cost_value(iter)-cost_value(iter-1))/cost_value(iter) < eps
        break
    end

end

end