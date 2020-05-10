%% main file of the code (demo)
% This code provides a simple demo on how to use the code. This is an 
% implementation of PREMA and B-PREMA algorithms in the paper titled:
% "PREMA: Principled Tensor Data Recovery from Multiple Aggregated Views"
% 
%
% To run this code, you need to download TensorLab package (https://www.tensorlab.net). 
% 
% Once you have all the files in the TensorLab package and the files in code 
% in the same folder, this file should run. 
%
%
%
% Ref. 1: Almutairi, F.M., Kanatsoulis, C.I., and Sidiropoulos, N.D., 
% "PREMA: Principled Tensor Data Recovery from Multiple Aggregated Views." 
% arXiv preprint arXiv:1910.12001, 2019.
%
% Ref. 2: Almutairi F.M., Kanatsoulis C.I., Sidiropoulos N.D., "Tendi: Tensor 
% Disaggregation from Multiple Coarse Views," In Proc. of The Pacific-Asia 
% Conference on Knowledge Discovery and Data Mining (PAKDD), Singapore, May 2020.
% 
% 
%
% Faisal M. Almutairi (almut012@umn.edu), University of Minnesota, Jan, 2020

clear; close all; clc;
% The data sample is the Fabric Softeners data set used in the two 
% references above. The source of this dataset is the James M. Kilts Center, 
% University of Chicago Booth School of Business, and it was downloaded from: 
% https://www.chicagobooth.edu/research/kilts/datasets/dominicks
load('Data_sample.mat')


% Generate two aggregated tensors, Yt and Yc, from tensor X: 
% 1) Matrix U is used to aggregate X over the first dimension (stores). U 
%    aggregates the 93 stores into 16 groups based on their location.
% 2) W is used to aggregate X over the third dimension (time in weeks). W 
%    aggregates the weeks into months (every 4 weeks). 
% 3) V = I (identity) in this example.
[Yt, Yc] = Generate_aggregate(X, U, V, W);
% Note that the aggregated tensors have NaN at the entries of missing data
% (read inside Generate_aggregate.m for aggregation details).

maskX = ~isnan(X); % mask of the indices of the observed entries
max_iter_init = 10; % number of iteration in the CPD step in the initialization algorithms 
max_iter = 10; % number of iterations for PREMA and PREMA-B
Rank = 40; % tensor rank
%% Oracle Baseline: CPD of the original tensor X
% CPD function in TensorLab package. We give it the total number of 
% iterations used by our models. 
[Ucpd, ~] = cpd(fmt(X), Rank, 'MaxIter', max_iter_init+max_iter); 
X_CPD = cpdgen(Ucpd); % reconstruct tensor X using the CPD factors A, B, abd C.
% Compute the Normalized Disaggregation Error (NDE) of X_CPF
NDE_CPD = (norm(X(maskX) - (X_CPD(maskX)), 2)^2)/norm(X(maskX), 2)^2;
%% %%%%%%%%%% PREMA %%%%%%%%%%%%%%%%%%% 
% 1) initialize PREMA using Initialization_of_PREMA.m (APPENDIX C in the paper)  
[A0, B0, C0] = Initialization_of_PREMA(Yt, Yc, U, V, W, Rank, max_iter_init);  
Fac0{1} = A0;
Fac0{2} = B0;
Fac0{3} = C0;
X_PREMA_init = cpdgen(Fac0); % reconstruct tensor X using the initial values of factors A, B, abd C.   
% Compute the NDE error of X_PREMA_init:
NDE_PREMA_init = (norm(X(maskX) - (X_PREMA_init(maskX)), 2)^2)/norm(X(maskX), 2)^2;

% 2) run the iterative steps of PREMA (Algorithm 1 in the paper)
[A, B, C, cost_PREMA] = PREMA_algorithm(Yt, Yc, U, V, W, 1, max_iter, A0, B0, C0);
Fac{1}=A;
Fac{2}=B;
Fac{3}=C;
X_PREMA = cpdgen(Fac); % reconstruct tensor X using the factors A, B, and C estimated by PREMA. 
% Compute the NDE error of X_PREMA
NDE_PREMA = (norm(X(maskX) - (X_PREMA(maskX)), 2)^2)/norm(X(maskX), 2)^2;
%% %%%%%%%%%% Blind PREMA (B-PREMA)%%%%%%%%%%%%%%%%%%% 
% 1) initialize B-PREMA using Initialization_of_Blind_PREMA.m  
[Ab0, A_tilde0, Bb0, Cb0, C_tilde0] = Initialization_of_Blind_PREMA(Yt, Yc, Rank, max_iter) ; 
FacB0{1} = Ab0;
FacB0{2} = Bb0;
FacB0{3} = Cb0;
X_BPREMA_init = cpdgen(FacB0); % reconstruct tensor X using the initial values of factors A, B, abd C.   
% Compute the NDE error of X_init:
NDE_BPREMA_init = (norm(X(maskX) - (X_BPREMA_init(maskX)), 2)^2)/norm(X(maskX), 2)^2;

% 2) run the iterative steps of B-PREMA (Algorithm 2 in the paper)
[Ab, Bb, Cb, cost_BPREMA] = Blind_PREMA_algorithm(Yt, Yc, 1, 100, max_iter, Ab0, A_tilde0, Bb0, Cb0, C_tilde0);
FacB{1} = Ab;
FacB{2} = Bb;
FacB{3} = Cb;
X_BPREMA = cpdgen(FacB); % reconstruct tensor X using the factors A, B, and C estimated by B-PREMA. 
% Compute the NDE error of X_B-PREMA
NDE_BPREMA = (norm(X(maskX) - (X_BPREMA(maskX)), 2)^2)/norm(X(maskX), 2)^2;   
%% Plot NDE of all methods
figure
names = {'PREMA', 'B-PREMA', 'CPD (oracle)'};
x = 1:3; 
y = [NDE_PREMA, NDE_BPREMA, NDE_CPD];                    
b = bar(x, y ,0.5,'LineWidth',1.5);
b.FaceColor = 'flat';
b.CData(1,:) = [228,26,28]/255;  
b.CData(2,:) = [55,126,184]/255;
b.CData(3,:) = [77,175,74]/255;

grid on
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',20,'FontWeight','bold')
set(gca,'xticklabel',names)
ylabel('NDE')