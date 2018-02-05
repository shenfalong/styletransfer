function im_input_pad=pad_image_16(im_input)  
  height = size(im_input,1);
  width = size(im_input,2);
  
  pad_height = mod(16 - mod(height,16),16);
  pad_width = mod(16 - mod(width,16),16);
  

  
  im_R=repmat(0,[height+pad_height width+pad_width]);
  im_G=repmat(0,[height+pad_height width+pad_width]);
  im_B=repmat(0,[height+pad_height width+pad_width]);

  im_input_pad=uint8(cat(3,im_R,im_G,im_B));
  
  im_input_pad(1:height,1:width,1:3)=uint8(im_input);
end