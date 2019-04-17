function [ ResultsLocalizeEmitters, emitterModel ] = getLocalizations( FMaxima, PSFsigma )
%localizeBoxWrapper Function to wrap in 
%   Detailed explanation goes here
im_list = FMaxima.emitterImages;
% if statement from Sam to fix a cell bug
if ~iscell(im_list)
    im_list = {im_list};
end 
boxCoords = FMaxima.boxCoords();

% need to initialize the emitter model
P = struct(...%phaseIdx=5
    'model','Gauss2DsMAP',... %The class name of the emitter model to use
    'estimator','Newton'...   %The estimation technique to use
    );
PP = struct(...
    'model','Gauss2DsMLE',... %The class name of the emitter model to use
    'estimator','Newton'...   %The estimation technique to use
    );
% was using 'Gauss2DsMLE', now trying ''Gauss2DsMAP'
emitterConstructor = str2func(P.model);
AlternateEConstructor = str2func(PP.model); % localize with MLE as well

size_cats = FMaxima.boxCoords.sizeCategories;
Nsize_cats = FMaxima.boxCoords.NsizeCategories;

emitterModel = cellmap(@(i) emitterConstructor(size_cats(:,i), PSFsigma), 1:Nsize_cats);
altEmitterModel = cellmap(@(i) AlternateEConstructor(size_cats(:,i), PSFsigma), 1:Nsize_cats);

%For each box size category we need to gather all the ROI and fit them
%with one of the n boxCoords emitter models.
N = boxCoords.NsizeCategories;
etheta= cell(1,N);
crlb = cell(1,N);
lpu = cell(1,N); % log posterior unormalized
% we are using the Uniformllh, replace with LLR!
%Uniformllh = cell(1,N);
% start relying on alternate estimates to improve accuracy
atheta= cell(1,N);
llr = cell(1,N);
llh = cell(1,N); % log likelihood ratio of MLE
lln = cell(1,N); % log likelihood of noise
%thetaInit = cell(1,N);
frameIdx = cell(1,N);
boxIdx = cell(1,N);
ModelHessian = cell(1,N);
Npixels = cell(1,N);
for n=1:N % its easier if we keep the cell array!
    idxs = boxCoords.sizeIndexes{n};
    boxIdx{n} = idxs;
    frameIdx{n} = boxCoords.boxFrameIdx(idxs);
    %scales = boxCoords.scaleSigmas(1,boxCoords.boxScaleIdx(idxs));
    %positions = 0.5+double(0*boxCoords.boxOrigin(:,idxs));
    %positions = 0.5+double(boxCoords.boxCenter(:,idxs) - boxCoords.boxOrigin(:,idxs));
    %thetaInit{n} = double([positions; zeros(2,numel(idxs)); scales]);

    %if emitterModel{n}.nParams==4
   %     thetaInit{n} = thetaInit{n}(1:4,:); %For the 4-parameter models don't include the sigma-scales.
    %end
    %                 [etheta{n}, crlb{n}, llh{n}] = obj.emitterModel{n}.estimate(im_list{n}, P.estimator, thetaInit{n});
    if isempty(im_list{n})
        etheta{n}=[];
        atheta{n}=[];
        crlb{n}=[];
        llh{n}=[];
        llr{n} = [];
        lpu{n} = [];
        %Uniformllh{n} = [];
        ModelHessian{n} = [];
    else
        [etheta{n}, crlb{n}, lpu{n}] = emitterModel{n}.estimate(im_list{n}, P.estimator);
        [atheta{n}, crlb{n}, llh{n}] = altEmitterModel{n}.estimate(im_list{n}, PP.estimator);
        lln{n} = altEmitterModel{n}.noiseBackgroundModelLLH(im_list{n});
        llr{n} = -2*(llh{n}-lln{n});
        ModelHessian{n} = emitterModel{n}.modelHessian(im_list{n}, etheta{n});
        Npixels{n} = prod(boxCoords.sizeCategories(:,n))*ones(length(llr{n}),1);
    end

end
ResultsLocalizeEmitters.rawTheta = etheta; %rawTheta is in cell-based format for easy re-use with the emitter Model
%ResultsLocalizeEmitters.thetaInit = thetaInit; %save the theta init for later debugging.
ResultsLocalizeEmitters.thetaHess = ModelHessian;

%Form localizations
etheta = [etheta{:}];
atheta = [atheta{:}];
crlb = [crlb{:}];
llh = vertcat(llh{:})';
lpu = vertcat(lpu{:})';
llr = vertcat(llr{:})';
Npixels = vertcat(Npixels{:})';
%Uniformllh = vertcat(Uniformllh{:})';
boxIdx = double([boxIdx{:}]);
frameIdx = double([frameIdx{:}]);
HessianMat = [];
for ii = 1:N
    HessianMat = cat(3,HessianMat,ModelHessian{ii});
end

% Convert Hessian to Standard Errors
SE = 0*crlb;
if ~isempty(SE)
    SE(1,:) = squeeze(-HessianMat(1,1,:));
    SE(2,:) = squeeze(-HessianMat(2,2,:));
    SE(3,:) = squeeze(-HessianMat(3,3,:));
    SE(4,:) = squeeze(-HessianMat(4,4,:));
    SE(5,:) = squeeze(-HessianMat(5,5,:));
end

E = [etheta; sqrt(1./SE); lpu; llr; boxIdx; frameIdx; Npixels]'; %internal emitter format
AE = [atheta; sqrt(crlb); llh; llr; boxIdx; frameIdx; Npixels]'; %internal alt-emitter format
shift = double(boxCoords.boxOrigin([2,1],boxIdx)')-1; %shift switches x/y since boxxer deals in row/col and locs are in x/y
E(:,1:2) = E(:,1:2) + shift; %correct for box coords
AE(:,1:2) = AE(:,1:2) + shift;

%Sorted emitters by frame Idx to maintain relationship with localizations
[~,sidx] = sort(E(:,end));
E = E(sidx,:);
AE = AE(sidx,:);

ResultsLocalizeEmitters.rawEmitters = E;
ResultsLocalizeEmitters.rawAltEmitters = AE;

end

