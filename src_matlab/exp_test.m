function result = exp_test(image_path)
run('~/mondrian/external/vlfeat-0.9.20/toolbox/vl_setup.m')

food_classify = load('food_classify.mat');
% load form workspace in python

im = imread(image_path);
scale_factor = 480/max(size(im));
im = imresize(im, scale_factor, 'bilinear');
descrs = food_classify.encodeFunc(food_classify.encoder,im);
descrs = vl_homkermap(descrs,1,'kchi2') ;
scores = food_classify.w' * descrs + food_classify.b;
if scores > 0
  pred = 1;
else
  pred = 2;
end

pred_label = {'food', 'non-food'};
result.pred = pred;
result.score = scores;
result.label = pred_label(pred);

