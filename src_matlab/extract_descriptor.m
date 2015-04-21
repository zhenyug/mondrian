function features = extract_descriptor(images, cacheDir)
%% Obtain all image descriptors and store in cache
chunk_size = 5000;% save to cache every 500 images
chunk_counter = 0;
chunk_id = 1;
STANDARD_SIZE = 480;
numImages = numel(images) ;
prog_bar = ProgressBar(numImages); 
features = {};
fprintf('start extracting denseSIFT') ;
for i = 1:numImages
  prog_bar.progress; 
  im = images{i};
  [w, h] = size(im);
  resize_factor = STANDARD_SIZE/max(w,h);
  im = imresize(im,resize_factor,'bilinear');
  fea = getDenseSIFT(im) ;
  images{i} = i;%free up the memory
  features{end+1} = fea;
  chunk_counter = chunk_counter + 1;
  if chunk_counter == chunk_size
      save(fullfile(cacheDir,['chuck_',num2str(chunk_id), '.mat']), ...
          'features','-v7.3');
      clear features
      features = {};
      chunk_counter = 0;
      chunk_id = chunk_id + 1;
  end
end
save(fullfile(cacheDir,['chuck_',num2str(chunk_id), '.mat']), ...
          'features','-v7.3');

