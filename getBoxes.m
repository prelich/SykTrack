function [ ResultsFilterMaxima ] = getBoxes( im, PSFsigma, scales )
%filterBoxWrapper: A Wrapper Function to perform Box Finding Fast for me
%   Author: PKR UNM June 2016

% get sizes
imsize = size(im);
framesize = imsize(1:2);
if length(imsize) > 2
    frameT = imsize(3);
else
    frameT = 1;
end

sigmas = repmat( PSFsigma.*scales, 2,1);
boxxer=Boxxer2D(framesize, sigmas);

% MODIFY FOR SCALE SPACE FILTERING
fframes = boxxer.filterLoG(im);
% get maxima
[R.rawMaxima, R.rawMaximaVals] = boxxer.scaleSpaceLoGMaxima(im, 5, 5);
R.rawMaxima([3,4],:) = R.rawMaxima([4,3],:); % flip [x y s t] to [x y t s] coords
R.filteredSumImage = sumImage2D(fframes(:,:,:));
% filter maxima with triangle threshold
Maxima = R;
MaximaPass = false(length(Maxima.rawMaximaVals),1);
% Make a threshold for each scale
for ii = 1:length(scales)
    MaximaIndex = Maxima.rawMaxima(4,:)==ii;
    if nnz(MaximaIndex) == 0
        continue;
    end
    smax = sort(Maxima.rawMaximaVals(MaximaIndex),1,'descend');
    [~,R.maximaThreshold(ii)] = triThres(smax);
    % threshold at each scale
    MaximaPass(MaximaIndex) = Maxima.rawMaximaVals(MaximaIndex) >= R.maximaThreshold(ii);
end

R.filter = MaximaPass;
R.maxima = Maxima.rawMaxima(:,R.filter);
R.maximaVals = Maxima.rawMaximaVals(R.filter);

% boxCoords is a BoxCoords object which organizes all the information about
% a group of boxes including scale and frame information.
R.boxCoords = boxxer.generateBoxCoords(R.maxima, frameT);
[R.emitterImages, R.emitterFrames] = R.boxCoords.makeROI(im);

% Assign variable
ResultsFilterMaxima = R;


% %% RE-IMPLEMENT ORIGINAL RPT FUNCTIONALITY TO INCORPORATE SCALE SPACE FILTERING!
% P = obj.checkParamsFindMaxima();
% obj.updateWaitbar(0,'Phase: Find Maxima');
% tic;
% 
% obj.initializeBoxxer(true); %force reset
% fframes = obj.getFilteredFrames();
% if obj.nScales == 1
%     [R.rawMaxima, R.rawMaximaVals] = obj.boxxer.enumerateImageMaxima(fframes, P.maximaNeighborhoodSize);
%     R.filteredSumImage = sumImage2D(fframes(:,:,1)); %use only first scale to make filtered image
% else
%     switch P.method
%         case 'LoG'
%             [R.rawMaxima, R.rawMaximaVals] = obj.boxxer.scaleSpaceLoGMaxima(obj.getFrames(), P.maximaNeighborhoodSize, P.scaleNeighborhoodSize);
%         case 'DoG'
%             [R.rawMaxima, R.rawMaximaVals] = obj.boxxer.scaleSpaceDoGMaxima(obj.getFrames(), P.maximaNeighborhoodSize, P.scaleNeighborhoodSize);
%         otherwise
%             error('RPT:findMaxima','Unknown filter method "%s"',P.method);
%     end
%     R.rawMaxima([3,4],:) = R.rawMaxima([4,3],:); % flip [x y s t] to [x y t s] coords
%     R.filteredSumImage = sumImage2D(fframes(:,:,:,1)); %use only first scale to make filtered image
% end
% R.rawMaximaImage = obj.computeMaximaImage(R.rawMaxima, R.rawMaximaVals);
% obj.ResultsFindMaxima = R;
% 
% obj.updateWaitbar(1);
% obj.times.findMaxima = toc;
% fprintf('Find Maxima Time: %.3fs\n',obj.times.findMaxima);
% obj.setPhase(3);


end

