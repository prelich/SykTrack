function [ logEvidence ] = UniformEvidence( data_in, npixels )
% UniformObservedInformation Function to evidence of a uniform noise model
% inputs: data_in, the model is just the mean photon count

ims = reshape(data_in,npixels,[]);
% theta_bg = mean(ims)';

% observed information is a scalar here since theta_bg is the only
% parameter
% ObInfo = data_k/theta_bg.^2

% prior: gamma(1.1,1.1/3)
% 1.1/3 = .3667

% ObservedI = sum(ims)'+0.1;
% hardcode prior bias into hessian
% ObservedI = npixels./(mean(ims)'+0.2727)+0.1;
% 
% EvDenom = sqrt(ObservedI/2/pi);
% % gamma hyper parameters from MAPPEL, hard coded
% hyper parameters re-set to ~Jeffrey's prior!
% beta = 0.001; %.3667;
% alpha = 0.5; %1.1;
beta = .3667;
alpha = 1.1;

beta1 = beta+npixels;
counts = sum(ims)';
alpha1 = alpha+counts;

logEvidence = alpha*log(beta)-alpha1*log(beta1)+gammaln(alpha1)-gammaln(alpha)-sum(gammaln(ims+1))';

end

