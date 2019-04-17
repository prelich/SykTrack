function [ ResultsFilterEmitters ] = getFilteredLocalizations( ResultsLocalizeEmitters, ResultsFilterMaxima, emitterModel, FilterParams )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
    Q = struct(...%phaseIdx=6
    'certVsNoiseModel', -1 ...
    );
else
    Q = FilterParams;
end

% if statements from Sam
if ~iscell(ResultsFilterMaxima.emitterImages)
    ResultsFilterMaxima.emitterImages = {ResultsFilterMaxima.emitterImages};
end

E = ResultsLocalizeEmitters.rawEmitters;
AE = ResultsLocalizeEmitters.rawAltEmitters;
S = ResultsLocalizeEmitters;
tic;

boxes = ResultsFilterMaxima.boxCoords(); % the box coords object
N = size(E,1); % number of raw localziations
Nmod = size(emitterModel,2);
filter = zeros(N,1); %true if we will keep the localizations
                
% filterDescriptions = {'MinIntensity', 'MinSigma', 'MaxSigma', 'MaxPositionSE', 'UniformModelComparison', 'NoiseModelComparison','OverlapDistance'};
filterDescriptions = {'NoiseModelComparison'};
filterStepCode = 1; % increment this after each test.  This indicated which filter method was repsonsible for filtering point
% populate log likelihoods of the different emitter models
LLH = ResultsLocalizeEmitters.rawEmitters(:,11);
% need to create image and model vectors
npixels = zeros(Nmod,1);
imdata = cell(Nmod,1);
imodel = cell(Nmod,1);
%UevFactor = cell(Nmod,1);
UlogEvidence = cell(Nmod,1);

% Add an indexing array to ignore non-existant points
LogIndScales = true(Nmod,1);
for n = 1:Nmod
    if isempty(ResultsFilterMaxima.emitterImages{n})
        LogIndScales(n) = false;
    end
end
IndScales = 1:Nmod;
IndScales = IndScales(LogIndScales);
    
for n = IndScales
    npixels(n) = prod(emitterModel{n}.imsize);
    imdata{n} = reshape(ResultsFilterMaxima.emitterImages{n},npixels(n),[]);
    imdata{n}(imdata{n} < 0.01) = 0.01; % make all 0 and negative values finite!
    model_images = emitterModel{n}.modelImage(ResultsLocalizeEmitters.rawTheta{n});
    imodel{n} = reshape(model_images,npixels(n),[]);
    % create observed information for Noise model
    UlogEvidence{n} = UniformEvidence( ResultsFilterMaxima.emitterImages{n}, npixels(n) );
end
% UevFactor = cell2mat(UevFactor);
UlogEvidence = cell2mat(UlogEvidence);
% modify the negative model hessian to scale for log I, log Bg, and log sigma
MevFactor = cell(Nmod,1);
for n = IndScales
    tempHess = -S.thetaHess{n};
    % get the evidence factor by taking the determinant of the hessian
    evFactor = zeros(size(tempHess,3),1);
    for ii = 1:length(evFactor)
        evFactor(ii) = (det(tempHess(:,:,ii))/(2*pi)^5);
    end
    MevFactor{n} = evFactor;
end
MevFactor = cell2mat(MevFactor);

% This filter is useless, omitting...
% Filter out really crappy stuff
% if ~isempty(Q.certVsUniformModel) && Q.certVsUniformModel>=0
%     pass = cell(1,1);
%     for n=1
%         alpha = Q.certVsUniformModel;
%         LLRstat = 2*(LLH-LLHuniform);
%         threshold = chi2inv(alpha,emitterModel{n}.nParams-1);
%         pass{n} = LLRstat > threshold;
%     end
%     pass = cell2mat(pass);
%     filter(~pass)=filterStepCode;
% end

filterStepCode = filterStepCode+1;
if ~isempty(Q.certVsNoiseModel) && Q.certVsNoiseModel>=0
    pass = cell(Nmod,1);
    for n=IndScales
        alpha = Q.certVsNoiseModel;
        LLR = sum(imdata{n} - imodel{n} + imdata{n}.*(log(imodel{n})-log(imdata{n})))';
        X2_CDF=@(k,x) gammainc(x/2,k/2);
        X2=-2*LLR;
        k = npixels - emitterModel{n}.nParams;
        pvalue=1-X2_CDF(k,X2);
        % obj.MinPValue<=pvalue
        pass{n}=alpha<=pvalue;
    end
    pass = cell2mat(pass(IndScales));
    filter(~pass)=filterStepCode;
end
% filtering out negative observed information determinants, e.g. fits are
% worthless!
filterStepCode = filterStepCode+1;
pass = cell(1,1);
for n = 1
    pass{1} = MevFactor>0;
end
pass = cell2mat(pass);
filter(~pass)=filterStepCode;
MevFactor = sqrt(MevFactor);
% filtering out negative diagonal elements of observed information as well
filterStepCode = filterStepCode+1;
pass = cell(1,1);
for n = 1
    pass{n} = ResultsLocalizeEmitters.rawEmitters(:,6).^2>0;
    pass{n} = pass{n} & ResultsLocalizeEmitters.rawEmitters(:,7).^2>0;
    pass{n} = pass{n} & ResultsLocalizeEmitters.rawEmitters(:,8).^2>0;
    pass{n} = pass{n} & ResultsLocalizeEmitters.rawEmitters(:,9).^2>0;
    pass{n} = pass{n} & ResultsLocalizeEmitters.rawEmitters(:,10).^2>0;
end
pass = cell2mat(pass);
filter(~pass) = filterStepCode;

S.filter = filter;
S.filterDescriptions = filterDescriptions;
S.emitters = E(~filter,:);
S.altemitters = AE(~filter,:);
evDenom =MevFactor(~filter);
S.logEvidence = [LLH(~filter)-log(evDenom) UlogEvidence(~filter)];
S.TrueFalse = [-log(1+exp(S.logEvidence(:,2)-S.logEvidence(:,1))) ...
    -log(1+exp(S.logEvidence(:,1)-S.logEvidence(:,2)))];
% add TrueFalse to emitters
S.emitters = [S.emitters S.logEvidence];
% move pixel count to end
S.emitters(:,[end-2:end]) = S.emitters(:,[end-1,end,end-2]);

%Use the filter to make the rawTheta for each sizeCategory into a selected theta within the
%size category
S.theta = cellmap(@(k) ResultsLocalizeEmitters.rawTheta{k}(:,~filter(boxes.scaleIndexes{k})), 1);

ResultsFilterEmitters = S;

end

