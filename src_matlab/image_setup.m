function [images, im_label] = image_setup(dataDir, labelDir)
% dataDir = '../data/archive01/standard/';
% labelDir = '../data/archive01/food_label.mat';

%iterate through all im_name and im_label to read all image and pair with
%im_label
label_info = load(labelDir);
im_name = cellstr(label_info.im_name);
im_label = label_info.im_label;
im_failread_index = [];
images = {};

for i = 1:length(im_label)
    im_n = im_name{i};
    im_dir = fullfile(dataDir, [im_n '.jpg']);
    if exist(im_dir, 'file')
        try
            im = imread(im_dir);
            images = [images, im];
        catch
            im_failread_index = [im_failread_index, i];
            display(['image fail to read: ' im_dir]);
        end
    else
        display(['fail to read image: ' im_dir]);
        im_failread_index = [im_failread_index, i];
    end
end

% delete the labels for the failed images
im_label(im_failread_index) = [];
    
end