clc;clear;close all;
caffe('reset');
load city_seg_colormap.mat;
load invmappings.mat;
list=importdata('/home/shen/Dpan/datasets/cityscapes/ImageSets/val.txt');
for i=1:numel(list)
  disp(i);
  origin_im=imread(['/home/shen/Dpan/datasets/cityscapes/JPEGImages/',list{i},'_leftImg8bit.png']);
  origin_im_pad=pad_image_16(origin_im);
  %% **********************************
  % the first 1.0x
  scoremap3=matcaffe_fcn(origin_im_pad);
  
  if 0% multi-scale image
    scoremap3_ = matcaffe_fcn(flip(origin_im_pad,2));
    scoremap3_=flip(scoremap3_,2);
    
    % the second 1.3x
    [height,width]=pad_edge_16(round(size(origin_im_pad,1)*1.2),round(size(origin_im_pad,2)*1.2));
    im=imresize(origin_im_pad,[height,width]);
    scoremap=matcaffe_fcn(im);
    scoremap_=matcaffe_fcn(flip(im,2));
    scoremap_=flip(scoremap_,2);
    
    
    scoremap1=imresize(scoremap,[size(scoremap3,1) size(scoremap3,2)],'bilinear');
    scoremap1_=imresize(scoremap_,[size(scoremap3,1) size(scoremap3,2)],'bilinear');
   
    
    % the third 0.7x
    [height,width]=pad_edge_16(round(size(origin_im_pad,1)*0.7), round(size(origin_im_pad,2)*0.7));
    im=imresize(origin_im_pad,[height,width]);
    scoremap=matcaffe_fcn(im);
    scoremap_=matcaffe_fcn(flip(im,2));
    scoremap_=flip(scoremap_,2);
    
    scoremap5=imresize(scoremap,[size(scoremap3,1) size(scoremap3,2)],'bilinear');
    scoremap5_=imresize(scoremap_,[size(scoremap3,1) size(scoremap3,2)],'bilinear');
   
    
    %% ************************* combine
    allmap=cat(4,scoremap1,scoremap1_,scoremap3,scoremap3_,scoremap5,scoremap5_);
  else
    allmap=cat(4,scoremap3);
  end
  

  [scoremap_max,~] = max(allmap,[],4);
  scoremap_mean = mean(allmap,4);
  scoremap = (scoremap_max + scoremap_mean) / 2;
  classmap = scoremap;



  [~,data]=max(classmap,[],3);   
  data=data-1;
  data=data(1:size(origin_im,1),1:size(origin_im,2),:);
 

  data = invmappings(data+1);
  data = uint8(data);
  imwrite(data,city_seg_colormap/255,['../res/val/' list{i} '.png']);
end
