function [ tm ] = makeTrackMovie( tracks, im, frametime )
%makeTrackMovie function to take a track array and movie of singles and make a
% movie
%   inputs: tracks cell array from SykTrack and single sequence frametime of sec/frame 

% r = {'t', 'x', 'y', 'I', 'bg', 'sigma', 'SE_x', 'SE_y', 'SE_I', 'SE_bg', 'SE_sigma','frame'};
% 
% pter = {'X','Y','I','bg','PSFsigma','X_SE','Y_SE','I_SE',...
%                'bg_SE','PSFsigma_SE','LLemitter','LLuni','BoxIdx','FrameIdx',...
%                'true','false'};

ptracks = cellmap(@(x) x(:,[14,1:10,14]), tracks);
tracks = cellmap(@(x) [x(:,1)*frametime x(:,2:end)], ptracks);

frameBounds = [min(cellfun(@(T) min(T(:,end)),tracks)), max(cellfun(@(T) max(T(:,end)),tracks))];
im2 = im(:,:,frameBounds(1):frameBounds(2));

tm = TrackMovie(tracks,im2);

end

