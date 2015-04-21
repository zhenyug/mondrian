function encoder = train_encoder(cacheDirDescr)
STANDARD_SIZE = 480;
numSamplesPerWord = 1000;
numWords = 256 ;
numPcaDimensions = 60 ;
load(fullfile(cacheDirDescr,'chuck_1.mat'));
numImages = numel(features);

%initialize encoder
encoder.type = 'fv' ;
encoder.subdivisions = zeros(4,0) ;
encoder.readImageFn = @imread ;
encoder.extractorFn = @getDenseSIFT ;
encoder.numWords = numWords ;
encoder.renormalize = false ;
encoder.geometricExtension = 'none' ;
encoder.whitening = false;
layouts = {'1x1', '2x2'} ;
for i = 1:numel(layouts)
  t = sscanf(layouts{i},'%dx%d') ;
  m = t(1) ;
  n = t(2) ;
  [x,y] = meshgrid(...
    linspace(0,1,n+1), ...
    linspace(0,1,m+1)) ;
  x1 = x(1:end-1,1:end-1) ;
  y1 = y(1:end-1,1:end-1) ;
  x2 = x(2:end,2:end) ;
  y2 = y(2:end,2:end) ;
  encoder.subdivisions = cat(2, encoder.subdivisions, ...
    [x1(:)' ;
     y1(:)' ;
     x2(:)' ;
     y2(:)'] ) ;
end

%% Step 0: obtain sample image descriptors
numDescrsPerImage = ceil(numWords * numSamplesPerWord / numImages);
w = STANDARD_SIZE;
h = STANDARD_SIZE;
for i = 1:numImages
    fea = features{i};
    randn('state',0) ;
    rand('state',0) ;
    sel = vl_colsubset(1:size(fea.descr,2), single(numDescrsPerImage)) ;
    descrs{i} = fea.descr(:,sel) ;
    frames{i} = fea.frame(:,sel) ;
    frames{i} = bsxfun(@times, bsxfun(@minus, frames{i}(1:2,:), [w;h]/2), 1./[w;h]) ;
end
descrs = cat(2, descrs{:}) ;
frames = cat(2, frames{:}) ;

%% Step 1 (optional): learn PCA projection
if numPcaDimensions < inf 
    fprintf('learning PCA rotation/projection\n') ;
    encoder.projectionCenter = mean(descrs,2) ;
    x = bsxfun(@minus, descrs, encoder.projectionCenter) ;
    X = x*x' / size(x,2) ;
    [V,D] = eig(X) ;
    d = diag(D) ;
    [d,perm] = sort(d,'descend') ;
    d = d + 0 * max(d) ;%no whitening
    m = min(numPcaDimensions, size(descrs,1)) ;
    V = V(:,perm) ;
    if encoder.whitening
        encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
    else
        encoder.projection = V(:,1:m)' ;
    end
    clear X V D d ;
else
  encoder.projection = 1 ;
  encoder.projectionCenter = 0 ;
end

descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;

%% Step 2 (optional): geometrically augment the features

descrs = extendDescriptorsWithGeometry(encoder.geometricExtension, frames, descrs) ;
%% Step 3: learn a GMM vocabulary
fprintf('learning GMM vocabulary');
dimension = size(descrs,1) ;
numDescriptors = size(descrs,2) ;

switch encoder.type

  case {'fv'} ;
    vl_twister('state', 1) ;

    v = var(descrs')' ;
    [encoder.means, encoder.covariances, encoder.priors] = ...
      vl_gmm(descrs, encoder.numWords, 'verbose', ...
             'Initialization', 'kmeans', ...
             'CovarianceBound', double(max(v)*0.0001), ...
             'NumRepetitions', 1) ;
end