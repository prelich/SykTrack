classdef PTrack < handle
    %Pablo Track Basic Coordinate Tracking class
    %   Bayesian Tracking with Filters and variable Box Sizes
    %   Peter Relich August 2017 and August 2019, UPENN
    %   Peter Relich June 2016, UNM
    %   Mark J. Olah (mjo@cs.unm.edu) 2015, UNM
    properties (Constant=true, Hidden=true)
        MAX_BIRTHS = 15000; %More than this many births and we start to have a memory problem now.
        % Must use the sparseC++ version for these cases
    end
    properties
        %Emitter data:
        Position; % This is an NXfeature matrix of coordinate positions and relevant information. 
        % The property below defines each column
        Col_id; % This will be a Map object so I can figure out which column 
        % has the information I need in case I change the inputs
        % Track Parameters.  
        % Everything is gone until I realize I need it again
        NumParams = 5; % number of parameters fit per emitter
        
        %Frame information
        FrameStart; % Start frame index
        FrameEnd; % End frame index
        
        % Track Parameters
        MaxFrameGap = 15; % Maximum frame gap to connect for Gap closing
        FrameLook = 10; % Frame Look Ahead and Back length for intermediate phase
        MissProb = 0.69; % probability of missing a localization
        FakePlausible = 0.5; % maximum plausibility for a spurious emitter/track
        MinEv = 0.01; % value for the evidence, birth death cost experiment if no prior data exists
        Beta_0 = 0.23; % initial beta value for the diffusion prior
        Alpha_0 = 3; % initial alpha value for the diffusion prior 
        MinAlpha = 3; % minimum alpha value for relaxation of posteriors
        TAlpha = 0.99; % minimum emitter probability for track survival
        % flag to use the Lambert function to reduce priors if they become to overinformative.
        Lambert = true;
        TPVal = 0.01; % final pvalue for LLR test on output tracks
        % Counts
        Nemitters; %Number of emitters
        Ndims; %Number of spatial dimensions (at present Ndims=2 is only valid value. 
        % Easily extendable to 3 at some future point.)
        Ntracks=0; %Number of produced tracks
        Nframes; %Total number of frames from first frame to last frame
        
        %Computed parameters
        Rho; % mean Particle Density [#particles/px]
                
        % Intermediate data
        FrameRef; % cell array size [1,Nframes].  For each frame holds a vector of row 
        % (emitter) indexes in that frame. 
        FilterFrameRef; % cell array size [1, Nframes].  Each frame holds a vector of row 
        % (filtered emitter) indexes
        SegStartFrameIdx; % frame index for a track start size:[Ntracks,1]
        SegEndFrameIdx; % frame index for a track end size:[Ntracks,1]
        SegStartLocIdx; % localization(emitter) index for a track start size:[Ntracks,1]
        SegEndLocIdx; % localization(emitter) index for a track end size:[Ntracks,1]
        TrackStartFrameIdx; % frame index for a track start size:[Ntracks,1]
        TrackEndFrameIdx; % frame index for a track end size:[Ntracks,1]
        TrackStartLocIdx; % localization(emitter) index for a track start size:[Ntracks,1]
        TrackEndLocIdx; % localization(emitter) index for a track end size:[Ntracks,1]+
        
        Faux; % Localizations that were removed prior to gap closing
        FauxSegs; % Track segments that were removed prior to gap closing
        FauxGroupSegs; % Groups of track segments that were removed prior to gap closing
        FauxTracks; % Tracks that were removed after gap closing
        FrameLinks; % particle indexing property (matrix: Ntracks x frames)
        FilterFrameLinks;
        SegLinks; % secondary matrix of track links
        FilterSegLinks;
        
        % final outputs
        TrackLinks; % final matrix of track links
        GapCost; %Final gap closing costs
        
        % container Maps
        HV_map; % Map object so I can keep track of columns
        Inter_map; % Map object with inter priors which use two models: diffusion with/without drift
    end
    
    properties (Transient = true)
       Ind; % Cell array of links tracking forward and backward F2F
       C_Ind; % Cell array of links that were the same for both F2F calls
       Hyper_V; % Cell array of hyper priors for emitter motion
       
       InterHyper_V; % Matrix of hyper priors for the intergap phase
       InterGrouping; % cell array of grouped track segments prior to filtering
    end
    
    methods
        %% Constructor
        function obj = PTrack(position)
            obj.Nemitters = size(position,1); %emitters are rows
            obj.Ndims = size(position,2); %dimensions are cols
            obj.Position = position;
            % function to map the identities of position columns
            obj.MapColumns; 
            obj.MapHypers;
            frameCol = obj.Col_id('FrameIdx');
            obj.FrameStart = min(obj.Position(:,frameCol));
            obj.FrameEnd = max(obj.Position(:,frameCol));
            obj.Nframes = obj.FrameEnd-obj.FrameStart+1;
            % associate the coordinates by order by frame
            obj.computeFrameRef();
            % calculate avg. particle density
            obj.computeParticleDensity();
            obj.Faux=false(size(position,1),1);
        end
        
        function MapColumns(obj)
           keyset = {'X','Y','I','bg','PSFsigma','X_SE','Y_SE','I_SE',...
               'bg_SE','PSFsigma_SE','LLemitter','LLR','BoxIdx','FrameIdx',...
               'emitter','fake','NPixels'};
           valueset = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17];
           obj.Col_id = containers.Map(keyset,valueset);
        end
        
        function MapHypers(obj)
           keyset = {'Vx','Gamx','Vy','Gamy','alpha','beta','alpha0','beta0'};
           valueset = [1,2,3,4,5,6,7,8];
           obj.HV_map = containers.Map(keyset,valueset);
           
           keyset = {'Vx','Gamx','Vy','Gamy','alpha','beta','alpha0','beta0','fake'};
           valuest = [1,2,3,4,5,6,7,8,9];
           obj.Inter_map = containers.Map(keyset,valuest);
        end
         
        %% highest level of tracking code
        function doLAP(obj)
            % intermediate variables of importance
            % perform rescaling for variables of interet
            FrameCol = obj.Col_id('FrameIdx');
            n = zeros(obj.FrameEnd-obj.FrameStart+1,1); % number of particles to link in a frame
            for tt = obj.FrameStart:obj.FrameEnd
                trel = tt-obj.FrameStart+1;
                n(trel) = nnz(obj.Position(:,FrameCol) == tt); % get # of coords for the frame;
            end
            % Initialize hyper priors
            obj.Hyper_V = initializeHypers(obj,n);
            obj.Ind=cell(obj.FrameEnd-obj.FrameStart,4); % particle link indices (col 1,2)
            % and costs (col 3,4)
            % track forward
            obj.trackFForward(n);
            % track backward
            obj.trackFBackward(n);
            % get non-conflict Indices
            obj.getNonConflictedInd(n);
            % intermediate gap closing phase
            [NTracks,newLinks]=obj.interTrack(n);       
            % filtering code!
            [FNTracks,~] = obj.filterBadShortTracks(newLinks,NTracks);
            % return if there are no tracks
            if nnz(obj.Faux) == length(obj.Faux)
                return;
            end
            % generate the appropriate inputs for gapclosing
            [Hypers, Coords] = obj.getGapInputs(obj.FilterSegLinks);
            % Gap Closing
            costGap = obj.GapClose(FNTracks,Hypers,Coords);
            [links12, links21, cost] = lap(costGap,0,0,0);            
            % export track data
            obj.GapCost = cost;
            obj.getFinalOutput(FNTracks, links12, links21, obj.FilterSegLinks);
            %% NEW CODE! Time to test!
            obj.trackThreshold;
        end
        
        %% lower level methods follow in the lines below
        %% populate hyper priors from frame 2 frame
        function OhyperV = initializeHypers(obj,n)
            % initialize cell arrays
            OhyperV = cell(length(n),2);
            for tt = 1:length(n)
                OhyperV{tt} = zeros(n(tt),6);
                % Hyper_V has parameters:
                % V_x gam_x V_y gam_y alpha beta
                HVc = values(obj.HV_map,{'Vx','Gamx','Vy','Gamy','alpha','beta','alpha0','beta0'});
                OhyperV{tt,1}(:,[HVc{1},HVc{3}]) = 0;
                OhyperV{tt,1}(:,[HVc{2},HVc{4}]) = 1 ;
                OhyperV{tt,1}(:,HVc{5}) = obj.Alpha_0;
                OhyperV{tt,1}(:,HVc{6}) = obj.Beta_0;
                OhyperV{tt,1}(:,HVc{7}) = obj.Alpha_0;
                OhyperV{tt,1}(:,HVc{8}) = obj.Beta_0;
                OhyperV{tt,2} = OhyperV{tt,1};
            end
        end
        
        function [OhyperV] = updateHypers(obj,tc,nc,prelinks,OhyperV)
            if isempty(prelinks)
                return; % no past, no problem!
            end
            % nc is the number of particles in the update frame and the
            % prior frame, where prior means the location we update from
            % tc is the update frame and the prior frame
            % prelinks is a link matrix that associates particles in tc(1)
            % to particles in tc(2)
            % check for a prior history
            HVc = values(obj.HV_map,{'Vx','Gamx','Vy','Gamy','alpha','beta','alpha0','beta0'});
            for kk = 1:nc(1)
                % history check
                pind = prelinks(kk,2);
                if pind < nc(2)
                    % grab old motion hypers
                    Vx = OhyperV{tc(2)}(pind,HVc{1});
                    Gamx = OhyperV{tc(2)}(pind,HVc{2});
                    Vy = OhyperV{tc(2)}(pind,HVc{3});
                    Gamy = OhyperV{tc(2)}(pind,HVc{4});
                    alpha = OhyperV{tc(2)}(pind,HVc{5});
                    beta = OhyperV{tc(2)}(pind,HVc{6});
                    alpha0 = OhyperV{tc(2)}(pind,HVc{7});
                    beta0 = OhyperV{tc(2)}(pind,HVc{8});
                    % grab the sufficient statistic, the linking distances!
                    priorc = obj.FrameRef{tc(2)}(pind);
                    currc = obj.FrameRef{tc(1)}(kk);
                    dx = (obj.Position(currc,obj.Col_id('X'))-obj.Position(priorc,obj.Col_id('X')));
                    varx = obj.Position(currc,obj.Col_id('X_SE'))^2+obj.Position(priorc,obj.Col_id('X_SE'))^2;
                    dy = (obj.Position(currc,obj.Col_id('Y'))-obj.Position(priorc,obj.Col_id('Y')));
                    vary = obj.Position(currc,obj.Col_id('Y_SE'))^2+obj.Position(priorc,obj.Col_id('Y_SE'))^2;
                    % check to see if new displacement was within expected
                    % entropy loss, lambert function is slow, add if
                    % statement to avoid calling it when possible!
                    Qx = 2*(1+Gamx);
                    Qy = 2*(1+Gamy);
                    dval = log(1 + ((dx-Vx)^2/Qx+(dy-Vy)^2/Qy)/(2*beta));
                    if dval > 1/alpha && obj.Lambert
                        funkshun = @(kapsilon) abs((log(kapsilon)+1+1/alpha)/(kapsilon*alpha+1) - dval);
                        ups = fminsearch(funkshun,1);
                        % relax prior variances if they fail entropy check
                        ups = min(1,ups); % only change priors if expected scaling is less than 1
                        % make sure uncertainty does not fall below initial
                        % priors!  Make alpha less than 1 since 1 information unit is added
                        % with the appended localization
                        Gfac = (obj.MinAlpha-1)/alpha;
                        ups = max(Gfac,ups);
                        Gamx = (1+Gamx)/ups-1;
                        Gamy = (1+Gamy)/ups-1;
                        alpha = alpha*ups;
                        beta = beta*ups;
                    end                    
                    % update the current priors
                    OhyperV{tc(1)}(kk,HVc{1}) = (Gamx*dx+Vx)/(Gamx+1);
                    OhyperV{tc(1)}(kk,HVc{2}) = Gamx/(Gamx+1);
                    OhyperV{tc(1)}(kk,HVc{3}) = (Gamy*dy+Vy)/(Gamy+1);
                    OhyperV{tc(1)}(kk,HVc{4}) = Gamy/(Gamy+1);
                    OhyperV{tc(1)}(kk,HVc{5}) = alpha+1;
                    OhyperV{tc(1)}(kk,HVc{6}) ...
                        = beta+(((dx-Vx)^2+varx)/(1+Gamx)+((dy-Vy)^2+vary)/(1+Gamy))/4; 
                    OhyperV{tc(1)}(kk,HVc{7}) = alpha0+1;
                    OhyperV{tc(1)}(kk,HVc{8}) ...
                        = beta0+(dx^2+varx+dy^2+vary)/4; 

                end
            end            
        end
        
        %% FRAME 2 FRAME higher level code SECTION
        function obj=trackFForward(obj,n)
            % loop through all the frames forward and run the frame 2 frame analysis
            for tt = obj.FrameStart:obj.FrameEnd-1
                % make a relative time variable
                trel = tt-obj.FrameStart+1;
                % initialize time and number vectors
                tcomp = [trel+1,trel];
                ncomp = [n(trel+1),n(trel)];
                costFrame = obj.Frame2Frame(tcomp,ncomp,obj.Hyper_V(:,1));
                % solve the LAP on the matrix if the matrix is not empty
                % otherwise skip, everything is empty for those frames
                if ~isempty(costFrame)
                    [links12, links21, cost] = lap(costFrame,0,0,0);
                    % store associations in the properties
                    obj.Ind{trel,1} = [links12, links21];
                    obj.Ind{trel,3} = cost;
                end
                prelinks = obj.Ind{trel,1};
                % update hyper priors for the next frame
                obj.Hyper_V(:,1)...
                    =obj.updateHypers(tcomp,ncomp,prelinks,obj.Hyper_V(:,1));
            end
        end
        
        function obj=trackFBackward(obj,n)
            % loop through all the frames backwards and run the frame 2 frame analysis
            for tt = obj.FrameEnd:-1:obj.FrameStart+1
                % make a relative time variable
                trel = tt-obj.FrameStart+1;
                % initialize time and number vectors
                tcomp = [trel-1,trel];
                ncomp = [n(trel-1),n(trel)];
                costFrame = obj.Frame2Frame(tcomp,ncomp,obj.Hyper_V(:,2));
                % solve the LAP on the matrix if the matrix is not empty
                % otherwise skip, everything is empty for those frames
                if ~isempty(costFrame)
                    [links12, links21, cost] = lap(costFrame,0,0,0);
                    % store associations in the properties
                    obj.Ind{trel-1,2} = [links21, links12];
                    obj.Ind{trel-1,4} = cost;
                end
                % I think this is the problem...
                %prelinks = obj.Ind{trel-1,2};
                prelinks = [links12, links21];
                % update hyper priors for the next frame
                obj.Hyper_V(:,2)...
                    =obj.updateHypers(tcomp,ncomp,prelinks,obj.Hyper_V(:,2));
            end
        end
        % get indices that were the same
        function obj=getNonConflictedInd(obj,n)
            conf = arrayfun(@(i) obj.Ind{i,1} ~= obj.Ind{i,2},(1:size(obj.Ind,1))'...
                ,'UniformOutput',false);
            obj.C_Ind = cell(size(obj.Ind,1),1);
            for ii = 1:size(obj.C_Ind,1)
                if isempty(obj.Ind{ii}) || all(all(~conf{ii}))
                    obj.C_Ind{ii} = obj.Ind{ii};
                    continue;
                end
                p = n(ii);
                m = n(ii+1);
                conflictsC1 = [find(conf{ii}(:,1)) obj.Ind{ii,1}(conf{ii}(:,1),1)];
                conflictsC2 = [find(conf{ii}(:,2)) obj.Ind{ii,1}(conf{ii}(:,2),2)];
                % re-arrange indices of conflicts to go to birth/death/junk
                c1ind = [conflictsC1(:,1)>p conflictsC1(:,2)>m];
                c2ind = [conflictsC2(:,1)>m conflictsC2(:,2)>p];
                conflictsC1(:,1) = [conflictsC1(c1ind(:,1),1); conflictsC1(~c1ind(:,1),1)];
                conflictsC1(:,2) = [conflictsC1(~c1ind(:,2),2); conflictsC1(c1ind(:,2),2)];
                conflictsC2(:,1) = [conflictsC2(c2ind(:,1),1); conflictsC2(~c2ind(:,1),1)];
                conflictsC2(:,2) = [conflictsC2(~c2ind(:,2),2); conflictsC2(c2ind(:,2),2)];
                obj.C_Ind{ii} = obj.Ind{ii,1}; % we will have to resolve costs later
                obj.C_Ind{ii}(conflictsC1(:,1),1) =  conflictsC1(:,2); % conflicts point to garbage
                obj.C_Ind{ii}(conflictsC2(:,1),2) =  conflictsC2(:,2);
            end
        end

        %% frame 2 frame lower level code
        function costFrame = Frame2Frame(obj,tc,nc,HyperV)
            % nc(1) number of particles in the tc(1) frame to connect to
            % nc(2) number of particles in the tc(2) frame to connect from
            % populate birth and death and connection matrices
            % added logic to deal with blank frames
            caseval = logical(nc(2))*2 + logical(nc(1));           
            switch caseval
                case 0 % there are no points in any frame, blank cost matrix
                    costFrame = {};
                case 1 % there are only points in the second frame, birth only matrix
                    costFrame = diag(ones(1,nc(1)));
                case 2 % there are only points in the first frame, death only matrix
                    costFrame = diag(ones(1,nc(2)));
                case 3 % there are points in both frames to connect
                    bM = obj.frameBirth(nc);
                    dM = obj.frameDeath(tc,nc,HyperV);
                    cM = obj.frameConnect(tc,nc,HyperV);
                    % Make sure that there are no negative values
                    minEv = min([min(min(cM)), min(min(dM)), min(min(bM))]) - eps; 
                    bM(bM~=0) = bM(bM~=0) - minEv;
                    dM(dM~=0) = dM(dM~=0) - minEv;
                    cM = cM - minEv;
                    % build junk LR matrix
                    jM = (cM>0)';
                    jM = jM*eps; % minimize impact of costs with a small number
                    % build output cost matrix
                    costFrame = [cM dM; bM jM];
            end
        end
               
        function bM = frameBirth(obj,nc)
            % Calculate expected minEv displacement given initial priors
            val = log(obj.Beta_0/obj.Alpha_0+eps)-(1+1/obj.Alpha_0)*log(obj.MinEv);
            bM = diag(val.*ones(nc(1),1));
        end
        
        function dM = frameDeath(obj,tc,nc,HyperV)
            % Calculate entropy of all priors
            alpha0 = HyperV{tc(2)}(:,obj.HV_map('alpha0'));
            beta0 = HyperV{tc(2)}(:,obj.HV_map('beta0'));
            val = log(beta0./alpha0+eps)-(1+1./alpha0)*log(obj.MinEv);
            dM = diag(val);
        end
        
        function evidenceM = frameConnect(obj,tc,nc,HyperV)
            % get coordinate positions
            nind = obj.FrameRef{tc(2)};
            mind = obj.FrameRef{tc(1)};
            % hard-coded for a 2-D system
            cval = values(obj.Col_id,{'X','Y','X_SE','Y_SE','emitter','fake'});
            Xn = obj.Position(nind,cval{1});
            Yn = obj.Position(nind,cval{2});
            truen = obj.Position(nind,cval{5});
            falsen = obj.Position(nind,cval{6});
            fn = 1./(1 + exp(truen-falsen));
            Xm = obj.Position(mind,cval{1});
            Ym = obj.Position(mind,cval{2});
            truem = obj.Position(mind,cval{5});
            falsem = obj.Position(mind,cval{6});
            fm = 1./(1+exp(truem-falsem));
            % Grab the motion priors!
            beta = HyperV{tc(2)}(:,obj.HV_map('beta'));
            alpha = HyperV{tc(2)}(:,obj.HV_map('alpha'));
            Vx = HyperV{tc(2)}(:,obj.HV_map('Vx'));
            Vy = HyperV{tc(2)}(:,obj.HV_map('Vy'));
            Gamx = HyperV{tc(2)}(:,obj.HV_map('Gamx'));
            Gamy = HyperV{tc(2)}(:,obj.HV_map('Gamy'));
            betaz = HyperV{tc(2)}(:,obj.HV_map('beta0'));
            alphaz = HyperV{tc(2)}(:,obj.HV_map('alpha0'));
            % Grab the emitter priors!
            evidenceM = zeros(nc(2),nc(1));
            log2pi = log(2*pi);
            Qx = 2*(1 + Gamx);
            Qy = 2*(1 + Gamy);
            for jj = 1:nc(1)
                % cost with a diffusion and drift particle
                bx = ((Xm(jj)-Xn-Vx).^2)./Qx/2;
                by = ((Ym(jj)-Yn-Vy).^2)./Qy/2;
                bxz = ((Xm(jj)-Xn).^2)/4;
                byz = ((Ym(jj)-Yn).^2)/4;
                evidenceM(:,jj) = -alpha.*log(beta+eps)-log(alpha+eps)...
                    + (alpha+1).*log(beta+bx+by) +.5*log(Qx.*Qy) ...
                    + log2pi - log(1-obj.FakePlausible*fn) - log(1-obj.FakePlausible*fm(jj));
               % cost with a purely diffusive particle
                    Scost = -alphaz.*log(betaz+eps)-log(alphaz+eps)...
                        +(alphaz+1).*log(betaz+bxz+byz) + log(2) ...
                        + log2pi - log(1-obj.FakePlausible*fn) - log(1-obj.FakePlausible*fm(jj));
                evidenceM(:,jj) = min(evidenceM(:,jj),Scost);
            end
        end
        
        %% MINI GAP CLOSING SECTION, HIGHER LEVEL
        function [NTracks,newlinks]=interTrack(obj,n)
            T = obj.FrameEnd-obj.FrameStart+1;
            interInd = cell(T,2);
            NTracks = obj.getNTracks(obj.C_Ind,n);
            % temporarily putting everything into object to see if it will
            % plot obj.links and the starts and stops
            [obj.FrameLinks, globStart, globEnd]=obj.linkMatrix(obj.FrameRef,obj.C_Ind,NTracks,n);
            obj.SegStartFrameIdx = globStart.FrameIdx;
            obj.SegEndFrameIdx = globEnd.FrameIdx;
            obj.SegStartLocIdx = globStart.LocIdx;
            obj.SegEndLocIdx = globEnd.LocIdx;
            tempStartLog = false(NTracks,T);
            tempEndLog = false(NTracks,T);
            [Coords, obj.InterHyper_V] = generateInterHypers(obj,globStart,globEnd);
            for ii = 1:T
                % generate a short frame segment to "Gap Close"
                minFrame = max(1,ii-obj.FrameLook);
                maxFrame = min(T,ii+obj.FrameLook);
                tempStartLog(:,ii) = globStart.FrameIdx >= minFrame & globStart.FrameIdx <= maxFrame;
                tempEndLog(:,ii) = globEnd.FrameIdx >= minFrame & globEnd.FrameIdx <= maxFrame;
                tempStart.FrameIdx = globStart.FrameIdx(tempStartLog(:,ii));
                tempStart.LocIdx = globStart.LocIdx(tempStartLog(:,ii));
                tempEnd.FrameIdx = globEnd.FrameIdx(tempEndLog(:,ii));
                tempEnd.LocIdx = globEnd.LocIdx(tempEndLog(:,ii));
                % parse coordinates and hyper priors
                tempCoords{1} = Coords{1}(tempEndLog(:,ii),:);
                tempCoords{2} = Coords{2}(tempStartLog(:,ii),:);
                tempHyperPriors{1} = obj.InterHyper_V{1}(tempEndLog(:,ii),:);
                tempHyperPriors{2} = obj.InterHyper_V{2}(tempStartLog(:,ii),:);
                % now do gap closing on the short segment
                caseval = any(tempEndLog(:,ii))*2 + any(tempStartLog(:,ii));
                switch caseval
                    case 0 % there are no points in any frame, blank cost matrix
                        costMat = {};
                    case 1 % there are only points in the second frame, birth only matrix
                        costMat = diag(ones(1,sum(tempStartLog(:,ii))));
                    case 2 % there are only points in the first frame, death only matrix
                        costMat = diag(ones(1,sum(tempEndLog(:,ii))));
                    case 3 % there are points in both frames to connect
                        costMat = obj.interGap(tempStart,tempEnd,tempCoords,tempHyperPriors);
                end
                if ~isempty(costMat)
                    [links12, links21, cost] = lap(costMat,0,0,0);
                    interInd{ii,1} = [links12 links21];
                    interInd{ii,2} = cost;
                end
            end
            % combine all the link indices and resolve conflicts
            newlinks = obj.resolveInterConflict(interInd,NTracks,globStart,globEnd,tempStartLog,tempEndLog);
        end
        
        function [FTracks,newerLinks]=filterBadShortTracks(obj,newLinks,NTracks)
            NNTracks = sum(newLinks(1:NTracks,2)>NTracks);
            % get the short track segments and grouped link matrix
            [obj.SegLinks, obj.InterGrouping] = ...
                obj.getSegLinks(NTracks,newLinks(:,1),NNTracks,...
                obj.FrameRef,obj.FrameLinks);
            Tracks=obj.makeTracksArray(obj.Position,NNTracks,obj.SegLinks);
            obj.FauxGroupSegs = false(length(Tracks),1);
            obj.FauxSegs = false(NTracks,1);
            numPos = size(obj.Position,1);
            TFcol = cell2mat(values(obj.Col_id,{'emitter','fake','BoxIdx'}));
            % make a box to loc container map
            box2loc = containers.Map(obj.Position(:,TFcol(3)),1:numPos);
            % loop over each track and flag as bad tracks are found
            for ii = 1:length(Tracks)
                tempTrue = Tracks{ii}(:,TFcol(1));
                tempFalse = Tracks{ii}(:,TFcol(2));
                % do a geometric mean of the error
                alpha = 1/(1+exp(sum(tempFalse-tempTrue)/length(tempTrue)));
                if length(tempTrue) == 1 % get rid of loners
                    FauxBox = (Tracks{ii}(:,TFcol(3)));
                    celldim = ones(length(FauxBox),1);
                    FauxCell = mat2cell(FauxBox,celldim);
                    FauxLoc = values(box2loc,FauxCell);
                    FauxLoc = [FauxLoc{:}];
                    obj.Faux(FauxLoc) = true;
                    obj.FauxGroupSegs(ii) = true;
                    bad_segs = obj.InterGrouping{ii};
                    obj.FauxSegs(bad_segs) = true;
                    continue;
                end
                
                if alpha<obj.TAlpha
                    FauxBox = (Tracks{ii}(:,TFcol(3)));
                    celldim = ones(length(FauxBox),1);
                    FauxCell = mat2cell(FauxBox,celldim);
                    FauxLoc = values(box2loc,FauxCell);
                    FauxLoc = [FauxLoc{:}];
                    obj.Faux(FauxLoc) = true;
                    obj.FauxGroupSegs(ii) = true;
                    bad_segs = obj.InterGrouping{ii};
                    obj.FauxSegs(bad_segs) = true;
                end
            end
            % return if all points are removed
            if nnz(obj.Faux) == length(obj.Faux)
                warning('All points were filtered out.  Exiting SykTrack without tracks');
                FTracks = [];
                newerLinks = [];
                return;
            end
            % make a filter frame index
            for ii = 1:obj.Nframes
                filterInd = obj.Faux(obj.FrameRef{ii});
                obj.FilterFrameRef{ii} = obj.FrameRef{ii}(~filterInd);
            end
            % remove bad Track Links
            obj.FilterSegLinks = obj.SegLinks(~obj.FauxGroupSegs,:);
            obj.FilterFrameLinks = obj.FrameLinks(~obj.FauxSegs,:);
            % fix the indexing for obj.FrameLinks
            for ii = 1:obj.Nframes
                tempC = obj.FilterFrameLinks(:,ii);
                indtemp = logical(tempC);
                tempind = tempC(indtemp);
                zz = 1:length(tempind);
                [~, I] = sort(tempind);
                tempind(I) = zz;
                obj.FilterFrameLinks(indtemp,ii) = tempind;
            end
            % Output track number is smaller now
            FTracks = sum(~obj.FauxGroupSegs);
            FtrackSegments = sum(~obj.FauxSegs);
            truncList = [find(~obj.FauxSegs),(1:sum(~obj.FauxSegs))'];
            newList = zeros(size(newLinks,1),1);
            newList(truncList(:,1)) = truncList(:,2) ;
            tempLinks = newLinks;
            tempLinks(newLinks(:,2),1) = newList;
            tempLinks(newLinks(:,1),2) = newList;
            tempLinks = tempLinks([~obj.FauxSegs false(length(obj.FauxSegs),1)],:);
            newerLinks = [tempLinks;0*tempLinks];
            % populate all the zeros with dummy values
            numvals1 = newerLinks(newerLinks(:,1)~=0,1);
            listnum = (1:2*FtrackSegments)';
            listnum(numvals1) = []; % remove numbers that were used.
            newerLinks(newerLinks(:,1)==0,1) = listnum(end:-1:1);
            newerLinks(newerLinks(:,1),2) = 1:2*FtrackSegments;
        end
        
        %% WINDOWED GAP CLOSING SECTION, LOWER LEVEL
        %% build the link matrix
        function [Links,TrackStart,TrackEnd]=linkMatrix(obj,FrameRef,Ind,births,sv)
            T = length(Ind)+1; % length of frames spanned
            % initialize links matrix and start and end indices
            Links = zeros(births,T);
            Links(1:sv(1),1) = 1:sv(1);
            TrackStart.FrameIdx = zeros(births,1);
            TrackEnd.FrameIdx = zeros(births,1);
            % initialize frame start at 1 and correct with the real frame
            % start time at the end of the function
            TrackStart.FrameIdx(1:sv(1)) = 1;
            mt = sv(1); % number of assigned tracks, grows with birth
            % build links matrix, and trackStartFrameIdx,trackEndFrameIdx vectors
            LinkDex = cell(T,1);
            LinkDex{1}=[(1:sv(1))' (1:sv(1))'];
            for tt = 1:T-1
                % skip the loop if there are no links
                if isempty(Ind{tt}) || length(Ind{tt}) == 1
                    continue;
                end
                % define intermediate variables
                links12 = Ind{tt}(:,1);
                links21 = Ind{tt}(:,2);
                % death counter or localization linkage
                [LinkDex, TrackEnd] = obj.deathorlink(LinkDex,sv,links12,tt,TrackEnd);
                % birth counter, linkage start
                [LinkDex, TrackStart, mt] = obj.birthorlink(LinkDex,sv,links21,mt,tt,TrackStart);
            end
            % populate links
            for tt = 1:T
                if isempty(LinkDex{tt})
                    continue
                end
                Links(LinkDex{tt}(:,1),tt) = LinkDex{tt}(:,2);
            end
            % get the starts and ends relative to the frameStart
            TrackEnd.FrameIdx(TrackEnd.FrameIdx==0) = T;
            %Use the pre-computed track endpoints to speed up the gap-closing
            %localization (row) index for the start of each track into the obj.position matrix, et. al.
            TrackStart.LocIdx = arrayfun(@(i) FrameRef{TrackStart.FrameIdx(i)}( Links(i,TrackStart.FrameIdx(i)) ), ...
                1:births)';
            %localization (row) index for the start of each track into the obj.position matrix, et. al.
            TrackEnd.LocIdx = arrayfun(@(i) FrameRef{TrackEnd.FrameIdx(i)}( Links(i,TrackEnd.FrameIdx(i)) ),...
                1:births)';
        end
        %% death counter or link
        function [LinkDex, TrackEnd] = deathorlink(~,LinkDex,n,links12,tt,TrackEnd)
            if isempty(LinkDex{tt})
                return; % can't do anything if there is nothing here
            end
            kkind = LinkDex{tt}(:,1)';
            % determine if there is a death or if a particle is linked
            for kk = kkind
                % death count
                relind = LinkDex{tt}(:,1)==kk;
                if links12(LinkDex{tt}(relind,2)) > n(tt+1)
                    %                     LinkDex{tt+1}(kk,2) = 0; % death
                    TrackEnd.FrameIdx(kk) = tt;
                    % link to next id
                else
                    LinkDex{tt+1} = [LinkDex{tt+1} ; kk, links12(LinkDex{tt}(relind,2))];
                end
            end
        end
        %% birth counter, start a link point
        function [LinkDex, TrackStart, mt] = birthorlink(~,LinkDex,n,links21,mt,tt,TrackStart)
            for bb = 1:n(tt+1)
                % congratulations, a birth!
                if links21(bb) > n(tt)
                    mt=mt+1; % increase birth count
                    LinkDex{tt+1} = [LinkDex{tt+1}; mt, bb];
                    TrackStart.FrameIdx(mt) = tt+1;
                end
            end
        end
        
        %% pre-allocate positions and hyper priors
        function [Coords, HyperPriors] = generateInterHypers(obj,glonStart,globEnd)
            lenprior = length(glonStart.FrameIdx); % globally starts and ends have the same count
            HyperPriors{1} = zeros(lenprior,size(obj.Inter_map,1));
            HyperPriors{2} = zeros(lenprior,size(obj.Inter_map,1));
            poscol = cell2mat(values(obj.Col_id,{'X','Y','FrameIdx','X_SE','Y_SE','emitter','fake'}));
            Coords = cell(2,1);
            Coords{2} = obj.Position(glonStart.LocIdx,poscol);
            Coords{1} = obj.Position(globEnd.LocIdx,poscol);
            % make a link vector of non-connected tracks
            nonLinks = (2*lenprior:-1:1)'; % nothing will connect!
            % pull out segmented links prior to all connections
            [TrackMatrix, ~] = ...
                obj.getSegLinks(lenprior,nonLinks,lenprior,...
                obj.FrameRef,obj.FrameLinks);
            Tracks=obj.makeTracksArray(obj.Position,lenprior,TrackMatrix);
            % map index for hyper prior locations
            tempind = cell2mat(values(obj.Inter_map,{'Vx','Gamx','Vy','Gamy',...
                    'alpha','beta','alpha0','beta0','fake'}));
            % loop over each track and calculate the appropriate hyper
            % priors
            for ii = 1:length(Tracks)
                % start all priors as barely informative
                % velocity estimated forwards and backwards as a 1x2 vector
                alpha = [obj.Alpha_0 obj.Alpha_0];
                beta = [obj.Beta_0 obj.Beta_0];
                mux = [0 0];
                muy = [0 0];
                gammax = [1 1];
                gammay = [1 1];
                alpha0 = obj.Alpha_0;
                beta0 = obj.Beta_0;
                dx = diff(Tracks{ii}(:,poscol(1)));
                dy = diff(Tracks{ii}(:,poscol(2)));
                dt = diff(Tracks{ii}(:,poscol(3)));
                varx = Tracks{ii}(:,poscol(4)).^2;
                vary = Tracks{ii}(:,poscol(5)).^2;
                logfake = sum(Tracks{ii}(:,poscol(7)));
                logemitter = sum(Tracks{ii}(:,poscol(6)));
                % These hyper priors will need to be
                % calculated slightly different than the frame to frame.
                for jj = 1:length(dx)
                    % incorporate a backwards vector to do the estimation
                    % reverse time!
                    kk = length(dx)-jj+1;
                    Qx = 2*[dt(jj)*(1+gammax(1)*dt(jj)) dt(kk)*(1+gammax(2)*dt(kk))];
                    Qy = 2*[dt(jj)*(1+gammay(1)*dt(jj)) dt(kk)*(1+gammay(2)*dt(kk))];
                    % check to see if new displacement was within expected
                    % entropy loss, use if statement to avoid calling
                    % lambertw whenever possible
                    dval(1) = log(1 + ((dx(jj)-mux(1))^2/Qx(1)+(dy(jj)-muy(1))^2/Qy(1))/(2*beta(1)));
                    dval(2) = log(1 + ((dx(kk)-mux(2))^2/Qx(2)+(dy(kk)-muy(2))^2/Qy(2))/(2*beta(2)));
                    funkshun = @(kapsilon,ii) abs((log(kapsilon)+1+1/alpha(ii))/(kapsilon*alpha(ii)+1) - dval(ii));
                    if dval(1) > 1/alpha(1) && obj.Lambert
                        funk1 = @(kapsilon) funkshun(kapsilon,1);
                        ups = fminsearch(funk1,1);
                        % relax prior variances if they fail entropy check
                        ups = min(1,ups); % only change priors if expected scaling is less than 1
                        % make sure uncertainty does not fall below initial
                        % priors! Make alpha less than 1 since 1 information unit is added
                        % with the appended localization
                        Gfac = (obj.MinAlpha-1)/alpha(1);
                        ups = max(Gfac,ups);
                        gammax(1) = ((1+gammax(1)*dt(jj))/ups-1)/dt(jj);
                        gammay(1) = ((1+gammay(1)*dt(jj))/ups-1)/dt(jj);
                        alpha(1) = alpha(1)*ups;
                        beta(1) = beta(1)*ups;
                    end
                    
                    if dval(2) > 1/alpha(2) && obj.Lambert
                        funk2 = @(kapsilon) funkshun(kapsilon,2);
                        ups = fminsearch(funk2,1);
                        % relax prior variances if they fail entropy check
                        ups = min(1,ups); % only change priors if expected scaling is less than 1
                        % make sure uncertainty does not fall below initial
                        % priors! Make alpha less than 1 since 1 information unit is added
                        % with the appended localization
                        Gfac = (obj.MinAlpha-1)/alpha(2);
                        ups = max(Gfac,ups);
                        gammax(2) = ((1+gammax(2)*dt(jj))/ups-1)/dt(jj);
                        gammay(2) = ((1+gammay(2)*dt(jj))/ups-1)/dt(jj);
                        alpha(2) = alpha(2)*ups;
                        beta(2) = beta(2)*ups;
                    end
                    vecind = [jj,kk];
                    for ll = 1:2
                        zz = vecind(ll);
                        alpha(ll) = alpha(ll) + 1;
                        beta(ll) = beta(ll) + ((dx(zz)-mux(ll)*dt(zz))^2+varx(zz))/(2*Qx(ll))+...
                            ((dy(zz)-muy(ll)*dt(zz))^2+vary(zz))/(2*Qy(ll));
                        mux(ll) = (dx(zz)*gammax(ll) + dt(zz)*mux(ll))/(gammax(ll)*dt(zz)+1);
                        muy(ll) = (dy(zz)*gammay(ll) + dt(zz)*muy(ll))/(gammay(ll)*dt(zz)+1);
                        gammax(ll) = gammax(ll)/(1+gammax(ll)*dt(zz));
                        gammay(ll) = gammay(ll)/(1+gammay(ll)*dt(zz));
                    end
                    alpha0 = alpha0+1;
                    beta0 = beta0+(dx(jj)^2+dy(jj)^2+varx(jj)+vary(jj))/4/dt(jj);
                end
                % compress the fake value to a geometric average
                fake = 1./(1+exp( (logemitter-logfake)/(length(dx)+1) ));
                beta0=beta0+varx(end)/4+vary(end)/4;
                beta= beta+varx(end)/4+vary(end)/4;
                % assign elements to hyper prior matrix!
                tempvec = [mux(1) gammax(1) muy(1) gammay(1) alpha(1) beta(1) alpha0 beta0 fake];
                HyperPriors{1}(ii,tempind) = tempvec;
                tempvec = [mux(2) gammax(2) muy(2) gammay(2) alpha(2) beta(2) alpha0 beta0 fake];
                HyperPriors{2}(ii,tempind) = tempvec;
            end
        end

        %% small region gap closing code
        function costMat = interGap(obj,tempStart,tempEnd,tempCoords,tempHyperPriors)
            if isempty(tempEnd.FrameIdx) || isempty(tempStart.FrameIdx)
                return; % skip call if no frame segments appear here
            end
            % build birth matrix
            bM = obj.interBirth(tempStart,tempHyperPriors);
            % build death matrix
            dM = obj.interDeath(tempEnd,tempHyperPriors);
            % build conenction matrix
            [Vm, Rm, Cm] = obj.interConnect(tempStart,tempEnd,tempCoords,tempHyperPriors);
            % Make sure that there are no negative values
            minEv = min([min(Vm), min(dM), min(bM)]);
            bM(bM~=0) = bM(bM~=0) - minEv + eps;
            dM(dM~=0) = dM(dM~=0) - minEv + eps;
            Vm(Vm~=0) = Vm(Vm~=0) - minEv + eps;
            % convert values to sparse
            cM = sparse(Rm,Cm,Vm,length(dM),length(bM));
            bM = sparse(1:length(bM),1:length(bM),bM);
            dM = sparse(1:length(dM),1:length(dM),dM);
            % build junk LR matrix
            jM = (cM>0)';
            jM = jM*eps; % minimize impact of costs with a small number
            % build output cost matrix
            % Note to self: I need to make the gap closing matrix a sparse
            % off of 3 vectors, value (v), row (r) and column (c).  For now
            % I'm using a full matrix to get things to work, but the cross
            % over will be essential!
            costMat = [cM dM; bM jM];
        end
        
        function bM = interBirth(obj,tempStart,HyperPriors)
            %stTmp = min(obj.MaxFrameGap,tempStart.FrameIdx-1); % put track starts at base 0
%             % Calculate loss at expected minEv from all priors
                beta_s = HyperPriors{2}(:,obj.Inter_map('beta0'));
                alpha_s = HyperPriors{2}(:,obj.Inter_map('alpha0'));
                loss = log(beta_s./alpha_s + eps) - (1+1./alpha_s)*log(obj.MinEv);
                loss(isinf(loss)) = -log(obj.MinEv);
                bM = loss;
        end
        
        function dM = interDeath(obj,tempEnd,HyperPriors)
            %enTmp = min(obj.MaxFrameGap,obj.FrameEnd - tempEnd.FrameIdx); % figure out gap lengths from track ends
%             % Calculate loss at expected minEv from all priors
                beta_e = HyperPriors{1}(:,obj.Inter_map('beta0'));
                alpha_e = HyperPriors{1}(:,obj.Inter_map('alpha0'));
                loss = log(beta_e./alpha_e + eps) - (1+1./alpha_e)*log(obj.MinEv);
                loss(isinf(loss)) = -log(obj.MinEv);
                dM = loss;
        end
        
        function [Vm, Rm, Cm] = interConnect(obj,tempStart,tempEnd,Coords,HyperPriors)
            %cM = zeros(length(tempEnd.FrameIdx),length(tempStart.FrameIdx)); % initialize connection matrix
            % initialize connection matrix as a sparse with 3 columns
            % pre-initialize with 2 N log N tracks
            preAlloc = ceil(2*length(tempStart.FrameIdx)*log(length(tempStart.FrameIdx)));
            Vm = zeros(preAlloc,1);
            Rm = zeros(preAlloc,1);
            Cm = zeros(preAlloc,1);
            counter = 1;
            % This is a double for loop for now
            log2pi = log(2*pi);
            poscol = [1,2]; % this was defined in generateInterHypers
            timecol = 3; % this was defined in generateInterHypers
%             varcol = [4,5]; % this was defined in generate interHypers
            hypercol = cell2mat(values(obj.Inter_map,...
                {'beta','alpha','Vx','Vy','Gamx','Gamy','beta0','alpha0','fake'}));
            for mm=1:length(tempStart.FrameIdx)
                %vectorized temporal cuttoffs
                dT = tempStart.FrameIdx(mm)-tempEnd.FrameIdx;
                temporally_feasible = dT>0 & dT<obj.MaxFrameGap;
                Ntemporally_feasible = sum(temporally_feasible);
                if Ntemporally_feasible == 0
                    continue;
                end
                %vectorized spatial cuttoffs
                sqdisp = (Coords{2}(mm*ones(Ntemporally_feasible,1),poscol)...
                    - Coords{1}(temporally_feasible,poscol)).^2;
                deltaT = (Coords{2}(mm*ones(Ntemporally_feasible,1),timecol)...
                    - Coords{1}(temporally_feasible,timecol));
                % get the relative diffusion evidence from a Jeffreys prior on diffusion
                Jevidence = 4*deltaT./(sum(sqdisp,2)+0.01); 
                spatially_feasible = Jevidence>=obj.MinEv;
                feasible = temporally_feasible;
                feasible(temporally_feasible) = spatially_feasible;
                if isempty(feasible)
                    continue;
                end
                deltaT = deltaT(spatially_feasible);
                % get coordinate positions for start frame
                Xs = Coords{2}(mm,poscol(1));
                Ys = Coords{2}(mm,poscol(2));
                Ts = Coords{2}(mm,timecol);
                % load backwards prior for track start
                beta_s = HyperPriors{2}(mm,hypercol(1))+eps;
                alpha_s = HyperPriors{2}(mm,hypercol(2))+eps;
                Vx_s = HyperPriors{2}(mm,hypercol(3));
                Vy_s = HyperPriors{2}(mm,hypercol(4));
                Gamx_s = HyperPriors{2}(mm,hypercol(5));
                Gamy_s = HyperPriors{2}(mm,hypercol(6));
                beta0_s = HyperPriors{2}(mm,hypercol(7))+eps;
                alpha0_s = HyperPriors{2}(mm,hypercol(8))+eps;
                fs = HyperPriors{2}(mm,hypercol(9));
                % load forward prior for track end
                nn = find(feasible); % removed for loop in favor of vectorization
                    Xe = Coords{1}(nn,poscol(1));
                    Ye = Coords{1}(nn,poscol(2));
                    Te = Coords{1}(nn,timecol);
                    beta_e = HyperPriors{1}(nn,hypercol(1))+eps;
                    alpha_e = HyperPriors{1}(nn,hypercol(2))+eps;
                    Vx_e = HyperPriors{1}(nn,hypercol(3));
                    Vy_e = HyperPriors{1}(nn,hypercol(4));
                    Gamx_e = HyperPriors{1}(nn,hypercol(5));
                    Gamy_e = HyperPriors{1}(nn,hypercol(6));
                    beta0_e = HyperPriors{1}(nn,hypercol(7))+eps;
                    alpha0_e = HyperPriors{1}(nn,hypercol(8))+eps;
                    fe = HyperPriors{1}(nn,hypercol(9));
                    % combine the hyper priors!
                    Gamx = (Gamx_s.*Gamx_e)./(Gamx_s+Gamx_e);
                    Gamy = (Gamy_s.*Gamy_e)./(Gamy_s+Gamy_e);
                    DelT = Ts-Te;
                    Vx = (Vx_e.*Gamx_s+Vx_s.*Gamx_e)./(Gamx_s+Gamx_e);
                    Vy = (Vy_e.*Gamy_s+Vy_s.*Gamy_e)./(Gamy_s+Gamy_e);
                    alpha = alpha_e+alpha_s+eps;
                    beta = beta_e+beta_s+eps;
                    Qx = 2*DelT.*(1+DelT.*Gamx);
                    Qy = 2*DelT.*(1+DelT.*Gamy);
                    bx = ((Xs-Xe-Vx.*DelT).^2)./Qx/2;
                    by = ((Ys-Ye-Vy.*DelT).^2)./Qy/2;
                    bxz = ((Xs-Xe).^2)/4./DelT;
                    byz = ((Ys-Ye).^2)/4./DelT;
                    alphaz = alpha0_e+alpha0_s+eps;
                    betaz = beta0_e+beta0_s+eps;
                    % cost with a drifting particle
                    Ecost = -alpha.*log(beta+eps)-log(alpha+eps)...
                        +(alpha+1).*log(beta+bx+by+eps) +.5*log(Qx.*Qy) ...
                        + log2pi - (deltaT-1)*log(obj.MissProb) ...
                        -log(1-obj.FakePlausible*fe)-log(1-obj.FakePlausible*fs);
                    % cost with a purely diffusive particle
                    Scost = -alphaz.*log(betaz+eps)-log(alphaz+eps)...
                        +(alphaz+1).*log(betaz+bxz+byz+eps) + log(2*DelT) ...
                        + log2pi - (deltaT-1)*log(obj.MissProb) ...
                        -log(1-obj.FakePlausible*fe)-log(1-obj.FakePlausible*fs);
                    cost = min(Scost,Ecost);
                    if cost == 0
                        cost = eps;
                    end
                    Vm(counter:counter+length(nn)-1) = cost;
                    Rm(counter:counter+length(nn)-1) = nn;
                    Cm(counter:counter+length(nn)-1) = mm;
                    counter = counter+length(nn);
            end
            % remove unused elements
            if counter < preAlloc+1
                Vm(counter:end) = [];
                Rm(counter:end) = [];
                Cm(counter:end) = [];
            end
        end
        
        %% conflict resolution code for the mini gap closing
        function reslinks = resolveInterConflict(~,interInd,NTracks,...
                globStart,globEnd,tempStartLog,tempEndLog)
            % purpose of this function is to make a new index array for the
            % final gap closing links matrix
            newlinks = int32(zeros(NTracks,2));
            % get tracks per frame
            T = size(interInd,1);
            % assign values to the links vector
            for ii = 1:T
                % maxFrame = min(T,ii+obj.FrameLook);
                enGlobInd = globEnd.FrameIdx == ii;
                stGlobInd = globStart.FrameIdx == ii;
                % Map local values to global values
                enGlob = find(tempEndLog(:,ii));
                stGlob = find(tempStartLog(:,ii));
                ennum = length(enGlob);
                stnum = length(stGlob);
                % create a mapping vector
                mapVec = zeros(stnum+ennum,2);
                mapVec(:,1) = [stGlob;0*enGlob];
                mapVec(:,2) = [enGlob;0*stGlob];
                % assign global end links
                if ennum
                    frameEnd2StartLoc = interInd{ii,1}(globEnd.FrameIdx(tempEndLog(:,ii))==ii,1);
                    newlinks(enGlobInd,1) = mapVec(frameEnd2StartLoc,1);
                end
                % assign global start links
                if stnum
                    frameStart2EndLoc = interInd{ii,1}(globStart.FrameIdx(tempStartLog(:,ii))==ii,2);
                    newlinks(stGlobInd,2) = mapVec(frameStart2EndLoc,2);
                end
            end
            % now remove conflicts with the back and forth directions
            rem1 = 0;
            while ~isempty(rem1)
                z1 = find(newlinks(:,1)~=0);
                z2 = find(newlinks(:,2)~=0);
                zz1 = newlinks(newlinks(z1,1),2) == 0;
                zz2 = newlinks(newlinks(z2,2),1) == 0;
                rem1 = z1(zz1);
                rem2 = z2(zz2);
                newlinks(rem1,1) = 0;
                newlinks(rem2,2) = 0;
            end
            % perform conflict resolution on the links
            [A1,I1]=sort(newlinks(:,1));
            [A2,I2]=sort(newlinks(:,2));
            %then find duplicates position:
            J1=find(diff(A1)==0);
            J2=find(diff(A2)==0);
            %Finally pick the numbers:
            JJ1=unique([J1(:)',J1(:)'+1]); % the unique is needed when there are more
            JJ2=unique([J2(:)',J2(:)'+1]); %than two duplicates.
            % zero out corresponding ends/starts that were double linked
            enconf = A2(JJ2);
            enconf = enconf(enconf~=0);
            stconf = A1(JJ1);
            stconf = stconf(stconf~=0);
            newlinks(stconf,2) = 0;
            newlinks(enconf,1) = 0;
            newlinks(I1(JJ1),1) = 0; % zero out all conflicts
            newlinks(I2(JJ2),2) = 0;
            % make sure all links feed into one another
            z1 = find(newlinks(:,1)~=0);
            c2 = newlinks(newlinks(z1,1),2);
            keepers = z1 == c2; % make sure links go back to one another!
            newlinks(z1(~keepers),1) = 0; % remove any conflicts here
            numvals = newlinks(z1(keepers));
            listnum = (1:2*NTracks)';
            listnum(numvals) = []; % remove numbers that were used.
            % put the new links into the output and put ordered junk in
            % where all the conflicts reside
            reslinks = int32(zeros(2*NTracks,2));
            reslinks(1:NTracks,1) = newlinks(:,1);
            reslinks(reslinks(:,1)==0,1) = listnum(end:-1:1);
            reslinks(reslinks(:,1),2) = 1:2*NTracks;
        end
        
        %% GAP CLOSING SECTION
        %% code to generate inputs for gapclosing
        function [Hypers, Coords] = getGapInputs(obj,FilteredTrackLinks)
            % also need to populate TrackStarts, TrackEnds
            filterGrouping = obj.InterGrouping(~obj.FauxGroupSegs);
            NTracks = size(FilteredTrackLinks,1);
            obj.TrackStartFrameIdx = zeros(NTracks,1);
            obj.TrackStartLocIdx = zeros(NTracks,1);
            obj.TrackEndFrameIdx = zeros(NTracks,1);
            obj.TrackEndLocIdx = zeros(NTracks,1);
            Hypers = cell(2,1);
            Hypers{1} = zeros(NTracks,size(obj.Inter_map,1));
            Hypers{2} = zeros(NTracks,size(obj.Inter_map,1));
            for mm = 1:NTracks
                mmSegLinks = FilteredTrackLinks(mm,:);
                FramePoints = find(mmSegLinks);
                obj.TrackStartFrameIdx(mm) = FramePoints(1);
                obj.TrackEndFrameIdx(mm) = FramePoints(end);
                obj.TrackStartLocIdx(mm) = mmSegLinks(FramePoints(1));
                obj.TrackEndLocIdx(mm) = mmSegLinks(FramePoints(end));
                filterInd = filterGrouping{mm};
                Hypers{1}(mm,:) = obj.InterHyper_V{1}(filterInd(end),:);
                Hypers{2}(mm,:) = obj.InterHyper_V{2}(filterInd(1),:);
            end
            % globally starts and ends have the same count
            poscol = cell2mat(values(obj.Col_id,{'X','Y','FrameIdx','X_SE','Y_SE'}));
            for mm=1:NTracks % get hyper prior array for start points
                stid = obj.TrackStartLocIdx(mm);
                Coords{2}(mm,:) = obj.Position(stid,poscol);

                enid = obj.TrackEndLocIdx(mm); % get the actual matrix index
                Coords{1}(mm,:) = obj.Position(enid,poscol);
            end
            tempind = cell2mat(values(obj.Inter_map,{'alpha0','beta0'}));
            % pull out segmented links prior to all connections
            Tracks=obj.makeTracksArray(obj.Position,NTracks,obj.FilterSegLinks);
            % use the entire connected set of track segments for the
            % diffusion only model
            for ii = 1:length(Tracks)
                % start all priors from original bias
                alpha0 = obj.Alpha_0;
                beta0 = obj.Beta_0;
                dx = diff(Tracks{ii}(:,poscol(1)));
                dy = diff(Tracks{ii}(:,poscol(2)));
                dt = diff(Tracks{ii}(:,poscol(3)));
                varx = Tracks{ii}(:,poscol(4)).^2;
                vary = Tracks{ii}(:,poscol(5)).^2;
                % need to calculate priors without the uninformative input
                % prior or I will double count those values when I combine
                % evidence.  These hyper priors will need to be
                % calculated slightly different than the frame to frame.
                for jj = 1:length(dx)
                    alpha0 = alpha0+1;
                    beta0 = beta0+(dx(jj)^2+dy(jj)^2+varx(jj)+vary(jj))/4/dt(jj);
                end
                beta0 = beta0+varx(end)/4+vary(end)/4;
                % assign elements to hyper prior matrix!
                tempvec = [alpha0 beta0];
                Hypers{1}(ii,tempind) = tempvec;
                Hypers{2}(ii,tempind) = tempvec;
            end
            
        end
        %% high level gap closing code
        function costGap = GapClose(obj,NTracks,Hypers,Coords)
            if NTracks == 0
                error('No track segments to connect.')
            end
            % build birth matrix
            bM = obj.gapBirth(Hypers);
            % build death matrix
            dM = obj.gapDeath(Hypers);
            % build conenction matrix
            [Vm, Rm, Cm] = obj.gapConnect(NTracks,Hypers,Coords);
            % Make sure that there are no negative values
            minEv = min([min(Vm), min(dM), min(bM)]);
            bM(bM~=0) = bM(bM~=0) - minEv + eps;
            dM(dM~=0) = dM(dM~=0) - minEv + eps;
            Vm(Vm~=0) = Vm(Vm~=0) - minEv + eps;
            % convert values to sparse
            cM = sparse(Rm,Cm,Vm,NTracks,NTracks);
            bM = sparse(1:length(bM),1:length(bM),bM);
            dM = sparse(1:length(dM),1:length(dM),dM);
            % build junk LR matrix
            jM = (cM>0)';
            jM = jM*eps; % minimize impact of costs with a small number
            % build output cost matrix
            % Note to self: I need to make the gap closing matrix a sparse
            % off of 3 vectors, value (v), row (r) and column (c).  For now
            % I'm using a full matrix to get things to work, but the cross
            % over will be essential!
            costGap = [cM dM; bM jM];
        end
        
        %% Lower Level Gap Closing Methods
        % birth matrix for gap closing
        function bM = gapBirth(obj,HyperPriors)
            %stTmp = obj.TrackStartFrameIdx-1; % put track starts at base 0
            % ratio of surviors, particle density drops
            beta_e = HyperPriors{2}(:,obj.Inter_map('beta0'));
            alpha_e = HyperPriors{2}(:,obj.Inter_map('alpha0'));
            loss = log(beta_e./alpha_e + eps) -(1+1./alpha_e)*log(obj.MinEv);
            loss(isinf(loss)) = -log(obj.MinEv);
            bM = loss; % force deaths
        end
        % death matrix for gap closing
        function dM = gapDeath(obj,HyperPriors)
            % determine frame size
            %enTmp = obj.Nframes - obj.TrackEndFrameIdx; % figure out gap lengths from track ends
            beta_e = HyperPriors{1}(:,obj.Inter_map('beta0'));
            alpha_e = HyperPriors{1}(:,obj.Inter_map('alpha0'));
            loss = log(beta_e./alpha_e + eps) -(1+1./alpha_e)*log(obj.MinEv);
            loss(isinf(loss)) = -log(obj.MinEv);
            dM = loss; % force deaths
        end
        
        function [Vm, Rm, Cm] = gapConnect(obj,NTracks,HyperPriors,Coords)
            % initialize connection matrix as a sparse with 3 columns
            % pre-initialize with 2 N log N tracks
            preAlloc = ceil(2*NTracks*log(NTracks));
            Vm = zeros(preAlloc,1);
            Rm = zeros(preAlloc,1);
            Cm = zeros(preAlloc,1);
            counter = 1;
            % This is a double for loop for now
            log2pi = log(2*pi);
            poscol = [1,2]; % this was defined in getGapInputs
            timecol = 3; % this was defined in getGapInputs
            hypercol = cell2mat(values(obj.Inter_map,{'beta','alpha','Vx','Vy',...
                'Gamx','Gamy','beta0','alpha0'}));
            for mm=1:NTracks
                %vectorized temporal cuttoffs
                dT = obj.TrackStartFrameIdx(mm)-obj.TrackEndFrameIdx;
                temporally_feasible = dT>0 & dT<obj.MaxFrameGap;
                Ntemporally_feasible = sum(temporally_feasible);
                if Ntemporally_feasible == 0
                    continue;
                end
                %vectorized spatial cuttoffs
                sqdisp = (Coords{2}(mm*ones(Ntemporally_feasible,1),poscol)...
                    - Coords{1}(temporally_feasible,poscol)).^2;
                deltaT = (Coords{2}(mm*ones(Ntemporally_feasible,1),timecol)...
                    - Coords{1}(temporally_feasible,timecol));
                % get the relative diffusion evidence from a Jeffreys prior on diffusion
                Jevidence = 4*deltaT./(sum(sqdisp,2)+0.01); % prevent infs
                spatially_feasible = Jevidence>=obj.MinEv;
                feasible = temporally_feasible;
                feasible(temporally_feasible) = spatially_feasible;
                if isempty(feasible)
                    continue;
                end
                deltaT = deltaT(spatially_feasible);
                % get coordinate positions for start frame
                Xs = Coords{2}(mm,poscol(1));
                Ys = Coords{2}(mm,poscol(2));
                Ts = Coords{2}(mm,timecol);
                % load prior for track start
                beta_s = HyperPriors{2}(mm,hypercol(1));
                alpha_s = HyperPriors{2}(mm,hypercol(2));
                Vx_s = HyperPriors{2}(mm,hypercol(3));
                Vy_s = HyperPriors{2}(mm,hypercol(4));
                Gamx_s = HyperPriors{2}(mm,hypercol(5));
                Gamy_s = HyperPriors{2}(mm,hypercol(6));
                beta0_s = HyperPriors{2}(mm,hypercol(7));
                alpha0_s = HyperPriors{2}(mm,hypercol(8));
                % removed for loop of feasible here in favor of vectorization
                nn = find(feasible);
                %compute costMat according to diffusion
                % load forward prior for track end
                Xe = Coords{1}(nn,poscol(1));
                Ye = Coords{1}(nn,poscol(2));
                Te = Coords{1}(nn,timecol);
                beta_e = HyperPriors{1}(nn,hypercol(1));
                alpha_e = HyperPriors{1}(nn,hypercol(2));
                Vx_e = HyperPriors{1}(nn,hypercol(3));
                Vy_e = HyperPriors{1}(nn,hypercol(4));
                Gamx_e = HyperPriors{1}(nn,hypercol(5));
                Gamy_e = HyperPriors{1}(nn,hypercol(6));
                beta0_e = HyperPriors{1}(nn,hypercol(7));
                alpha0_e = HyperPriors{1}(nn,hypercol(8));
                % combine the hyper priors!
                Gamx = (Gamx_s.*Gamx_e)./(Gamx_s+Gamx_e);
                Gamy = (Gamy_s.*Gamy_e)./(Gamy_s+Gamy_e);
                DelT = Ts-Te;
                Vx = (Vx_e.*Gamx_s+Vx_s.*Gamx_e)./(Gamx_s+Gamx_e);
                Vy = (Vy_e.*Gamy_s+Vy_s.*Gamy_e)./(Gamy_s+Gamy_e);
                alpha = alpha_e+alpha_s+eps;
                beta = beta_e+beta_s+eps;
                Qx = 2*DelT.*(1+DelT.*Gamx);
                Qy = 2*DelT.*(1+DelT.*Gamy);
                bx = ((Xs-Xe-Vx.*DelT).^2)./Qx/2;
                by = ((Ys-Ye-Vy.*DelT).^2)./Qy/2;
                alphaz = alpha0_e+alpha0_s;
                betaz = beta0_e+beta0_s;
                bxz = ((Xs-Xe).^2)/4./DelT;
                byz = ((Ys-Ye).^2)/4./DelT;
                Ecost = -alpha.*log(beta+eps)-log(alpha+eps)...
                    +(alpha+1).*log(beta+bx+by+eps) +.5*log(Qx.*Qy) ...
                    + log2pi - (deltaT-1)*log(obj.MissProb);
                Scost = -alphaz.*log(betaz+eps)-log(alphaz+eps)...
                    +(alphaz+1).*log(betaz+bxz+byz+eps) + log(2*DelT) ...
                    + log2pi - (deltaT-1)*log(obj.MissProb);
                cost = min(Ecost,Scost);
                if cost == 0
                    cost = eps;
                end
                Vm(counter:counter+length(nn)-1) = cost;
                Rm(counter:counter+length(nn)-1) = nn;
                Cm(counter:counter+length(nn)-1) = mm;
                counter = counter+length(nn);
            end
            % remove unused elements
            if counter < preAlloc+1
                Vm(counter:end) = [];
                Rm(counter:end) = [];
                Cm(counter:end) = [];
            end
        end
        %% LLR track filtering
        function obj = trackThreshold(obj)
            % Method to remove tracks based on LLR test
            Tracks=obj.makeTracksArray(obj.Position,obj.Ntracks,obj.TrackLinks);
            obj.FauxTracks = false(length(Tracks),1);
            LLRcol = cell2mat(values(obj.Col_id,{'LLR'}));
            TFcol = cell2mat(values(obj.Col_id,{'emitter','fake'}));
            NPixCol = cell2mat(values(obj.Col_id,{'NPixels'}));
            % assign the Xi2 function
            X2_CDF=@(k,x)gammainc(x/2,k/2);
            % loop over each track and flag as bad tracks are found
            for ii = 1:length(Tracks)
                tempLLR = Tracks{ii}(:,LLRcol);
                k = Tracks{ii}(:,NPixCol)-obj.NumParams;
                % remove individual awful fits
                X2 = 1-X2_CDF(k,tempLLR);
                pass = obj.TPVal < X2;
                tempLLR = tempLLR(pass);
                if length(tempLLR) < 2
                    obj.FauxTracks(ii) = true;
                    continue; % need at least 2 good fits per track
                end
                
                emitEvidence = Tracks{ii}(:,TFcol(1));
                emitEvidence = emitEvidence(pass);
                noisEvidence = Tracks{ii}(:,TFcol(2));
                noisEvidence = noisEvidence(pass);
                probVal = 1/(1 + exp(sum(noisEvidence-emitEvidence)/sum(pass)));
                if probVal < obj.TAlpha
                    obj.FauxTracks(ii) = true;
                end
            end
        end
                        
        %% get Ntracks
        function NTracks = getNTracks(~,Ind,n)
            NTracks = n(1);
            for ii = 1:(length(n)-1)
                if isempty(Ind{ii})
                    continue;
                end
                links21 = Ind{ii}(:,2);
                NTracks = NTracks + sum(links21(1:n(ii+1))>n(ii));
            end
        end
        
        %% Associate matrix coordinates with each frame for LAP calls
        function computeFrameRef(obj)
            T = obj.FrameEnd-obj.FrameStart+1; % number of frames to span
            FrameCol = obj.Col_id('FrameIdx');
            obj.FrameRef = cellmap(@(i) find(obj.Position(:,FrameCol)...
                == i+obj.FrameStart-1),1:T);
        end
      
        %% output formatting code
        function getFinalOutput(obj,Nsegments,links12,links21,SegLinks)
            % determine number of tracks
            obj.Ntracks = sum(links21(1:Nsegments)>Nsegments);
            % get final track info
            obj.TrackLinks = obj.getTrackLinks(Nsegments,links12,...
                obj.Ntracks,SegLinks);
        end
        
        function [TrackLinks, st_grouping] = ...
                getTrackLinks(obj,Nsegments,links12,Ntracks,SegLinks)
            % takes the gap closing links and builds a final association
            % matrix for the original input matrix (TrackLinks)
            st_grouping = obj.groupShortTracks(Nsegments,links12,Ntracks);
            % build the full track link matrix prior to frame ref
            % association
            c_links = zeros(Ntracks,obj.Nframes);
            for nn = 1:Ntracks
                for mm = st_grouping{nn}
                    c_links(nn,:) = c_links(nn,:)+SegLinks(mm,:);
                end
            end
            TrackLinks = c_links;
        end
        
        function getInterOutput(obj,Nsegments,links12,links21,FrameRef,FrameLinks)
            % determine number of tracks
            obj.Ntracks = sum(links21(1:Nsegments)>Nsegments);
            % get final track info
            obj.TrackLinks = obj.getSegLinks(Nsegments,links12,...
                obj.Ntracks,FrameRef,FrameLinks);
        end
        
        function [SegLinks, st_grouping] = ...
                getSegLinks(obj,Nsegments,links12,Ntracks,FrameRef,FrameLinks)
            % takes the gap closing links and builds a final association
            % matrix for the original input matrix (TrackLinks)
            SegLinks=zeros(Ntracks,obj.Nframes); % build the TrackLinks matrix
            st_grouping = obj.groupShortTracks(Nsegments,links12,Ntracks);
            % build the full track link matrix prior to frame ref
            % association
            c_links = zeros(Ntracks,obj.Nframes);
            for nn = 1:Ntracks
                for mm = st_grouping{nn}
                    c_links(nn,:) = c_links(nn,:)+FrameLinks(mm,:);
                end
            end
            % assign coordinate locations of tracks to the final link matrix
            for tt = 1:obj.Nframes
                tempc_links = c_links(:,tt);
                link_logical = tempc_links>0; % so we index track links properly!
                tempc_links = tempc_links(tempc_links>0); % so we don't index 0's!
                SegLinks(link_logical,tt) = FrameRef{tt}(tempc_links);
            end
        end
        
        function st_grouping = groupShortTracks(~,Nsegments,links12,Ntracks)
            % group the assosciations of short tracks into separate cell
            % arrays for compressing the intermediate link matrix into NtracksxT size
            st_grouping{Ntracks} = [];
            % group short tracks together
            count = 1; % start a counter of the grouping matrix
            remvec = true(Nsegments,1);
            id = 1; %remvec(1); % start with an id at 1
            for nn = 1:Nsegments
                if id > Nsegments
                    count = count+1;
                    id = find(remvec,1);
                end
                st_grouping{count} = [st_grouping{count} id];
                remvec(id) = false; % remove chosen id from vector
                id = links12(id);
            end
        end
        
        function Tracks=makeTracksArray(obj,L,NTracks,TrackLinks)
            if nargin < 3
                NTracks = obj.Ntracks;
            end
            Tracks{NTracks} = []; % initialize cell array
            % loop through cell array indices
            for nn = 1:NTracks
                % output format of cell array
                % 'x', 'y', 'lambda', 'I', 'std_x', 'std_y', 'std_lambda', 'std_I', 'frame'
                %find all links for track nn
                TrackLinksnn = TrackLinks(nn,:);
                TrackLinksnn = TrackLinksnn(TrackLinksnn>0); % remove 0 indices (intermittent gaps)
                Tracks{nn} = L(TrackLinksnn,:);
            end
        end
        
        %% calculate the average density of the coordinates
        function obj=computeParticleDensity(obj)
            frameCol = obj.Col_id('FrameIdx');
            Xcol = obj.Col_id('X');
            Ycol = obj.Col_id('Y');
            count = histc(obj.Position(:,frameCol), obj.FrameStart:obj.FrameEnd);
            ROIarea = zeros(length(count),1);
            % estimate a square region of the particles given max_min
            % position for each frame
            for ii = obj.FrameStart:obj.FrameEnd
                zz = ii-obj.FrameStart+1;
                if count(zz) == 0
                    continue;
                end
                Xframe = obj.Position(obj.Position(:,frameCol)==ii,Xcol);
                Yframe = obj.Position(obj.Position(:,frameCol)==ii,Ycol);
                ROIx = [max(Xframe) min(Xframe)];
                ROIy = [max(Yframe) min(Yframe)];
                ROIarea(zz) = diff(ROIx)*diff(ROIy);
            end
            % remove frames without localizations -- they won't be tracked
            obj.Rho = mean(count(ROIarea>0)./ROIarea(ROIarea>0)); % estimate density counts/pixel
        end
    end
    
    methods (Static = true)
        function tracks = tracks2RPTformat(ptracks,frametime)
            % converts Peters tracking output tracks into RPT track format.
            % RPT track column format = {'t', 'x', 'y', 'I', 'bg', 'sigma', 
            % 'SE_x', 'SE_y', 'SE_I', 'SE_bg', 'SE_sigma','frame'};
            %
            % Peter track column format = {'X','Y','I','bg','PSFsigma','X_SE','Y_SE','I_SE',...
            %                'bg_SE','PSFsigma_SE','LLemitter','LLuni','BoxIdx','FrameIdx',...
            %                'true','false'};
            if nargin <2
                frametime = 0.1018;
            end
            tracks1 = cellmap(@(x) x(:,[14,1:10,14]), ptracks);
            tracks = cellmap(@(x) [x(:,1)*frametime x(:,2:end)], tracks1);
        end
 
    end
end

