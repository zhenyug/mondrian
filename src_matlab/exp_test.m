function pred = exp_test(image_path)
run('../external/vlfeat-0.9.20/toolbox/vl_setup.m')

food_classify = load('food_classify.mat');
im = imread(image_path);
scale_factor = 480/max(size(im));
im = imresize(im, scale_factor, 'bilinear');
descrs = food_classify.encodeFunc(food_classify.encoder,im);
descrs = vl_homkermap(descrs,1,'kchi2') ;
scores = food_classify.w' * descrs + food_classify.b;
pred = scores > 0;
pred_label = {'food', 'non-food'};
if pred == 1
    fprintf('this image is %s\n with a score: %.2f\n', pred_label{1}, scores);
else
    fprintf('this image is %s\n with a score: %.2f\n', pred_label{2}, scores);
end
imshow(im);


