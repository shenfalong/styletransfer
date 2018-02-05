function  scoremap = matcaffe_fcn(im_input)

prototxt_file = 'prototxt/deploy_resnet_101_16x_v0.prototxt';
weights_file = 'caffemodel/resnet101_voc12_0_iter_80000_trainval_2GPU0.caffemodel';
if caffe('is_initialized') < 1
    if exist(prototxt_file, 'file') == 0
        % NOTE: you'll have to get the pre-trained ILSVRC network
        error('You need a prototxt file');
    end
    if ~exist(weights_file,'file')
        % NOTE: you'll have to get network definition
        error('You need the weight file');
    end
    caffe('init', prototxt_file, weights_file);
    caffe('set_device',0);
end


im_input = single(im_input(:,:,[3 2 1]));%rgb --> bgr

im_input(:,:,1) = im_input(:,:,1);%- 104.008;
im_input(:,:,2) = im_input(:,:,2);%- 116.669;
im_input(:,:,3) = im_input(:,:,3);%- 122.675;

images(:,:,:,1)=single(permute(im_input,[2 1 3]));%height x width --> width x height


input_data = {images};



tic
scores = caffe('forward',0, input_data);
toc

scoremap = permute(scores{1},[2 1 3]);

end
