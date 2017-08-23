function newImage = ChangeHairColor2(a,b,c,d)
    clc;
    close all;
    %clearvars;
    faceDetector=vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
    
    fname = strcat('images/', a);
    images=imread(fname);
    fname = strcat('out/', a);    
        copy=images;
 
        Chrom=rgb2ycbcr(copy);
        [rows columns numberOfColorChannels]=size(images);
        
        for i=1:size(Chrom,1)%Read row pixels
            for j=1:size(Chrom,2)%Read column pixels
                %This conditions will determine the face based on RGB colour value
                if (5<Chrom(i,j,1)<117 && 110<Chrom(i,j,2)<113 && Chrom(i,j,3)>128.2)
                    copy(i,j,1)=0;
                    copy(i,j,2)=0;
                    copy(i,j,3)=0;
                    %img = 255 * uint8(copy(i,j,3));
                    %If above conditions is not meet, other colour values will be set to 255 (white)
                else
                     copy(i,j,1)=255;
                     copy(i,j,2)=255;
                     copy(i,j,3)=255;

                end
            end
         end
        images2 = imcomplement(im2bw(255 * uint8(copy)));
        images2=imfill(images2,'holes'); 
        images2=bwareaopen(images2,200);
       
        
        redMatrix = images(:,:,1);
        greenMatrix = images(:,:,2);
        blueMatrix = images(:,:,3);
        redMatrix2 = images(:,:,1);
        J(:,:,1) = redMatrix < 100 ;
        J(:,:,2) = greenMatrix > 150;
        J(:,:,3) = blueMatrix >110;

        redMatrix2 = images(:,:,1);
        K(:,:,1) = redMatrix < 77 ;
      
        
        J1=imsubtract(255 * uint8(J(:,:,1)),255 * uint8(J(:,:,3)));
        J1face = bwareaopen(J1, 100);      
        J1face=imfill(J1face,'holes');  
        H = fspecial('average');
        woFJ(:,:,3) = bwareaopen(J(:,:,3), 4000);        
        
        K3=imsubtract(J1face,K);
        K3=bwareaopen(K3,20000);
      
        imageAdd=imadd(J1face,images2);      
        imageAdd=bwareaopen(imageAdd,10);
               
        J2=imsubtract(imageAdd,J1face); 
        J2 = bwareaopen(J2, 10000);
        J2=imadd(J2,K3);
        J2=imfill(J2,'holes');

        J2=im2bw(J2,.05)
        J2=imfill(J2,'holes');
        
        edgewoFJ(:,:,3) = edge(woFJ(:,:,3),'Sobel');      
        imageEdge=imadd(255 * uint8(J1face(:,:,1)),255 * uint8(edgewoFJ(:,:,3)));        
    
        J2Edge=imsubtract(imageEdge,255 * uint8(J2));
        J2Edge=imfill(imgaussfilt(J2Edge,.7),'holes');     

        
        Jbw=bwareaopen(J2Edge,3000);

        
        edge2 = edge(Jbw,'Sobel');    
        edge2=imgaussfilt(255 * uint8(edge2),.5);    
        
        edge2= imsubtract(255 * uint8(Jbw),255 * uint8(edge2))
        edge2=bwareaopen(im2bw(edge2,.05),100);
        
        %mask the original image with the masked hair
        if numberOfColorChannels == 1
            maskedImage=images; % Initialize with the entire image.
            maskedImage(~edge2) = 0; % Zero image outside the circle mask.
        else
            % Mask the image.
            maskedImage = bsxfun(@times, images, cast(edge2,class(images)));
        end
     
        %%%%%%%%%%%%%%%%%%%image with transparency
        origHair=maskedImage;      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % If it's grayscale, convert to color
        if numberOfColorChannels < 3
            rgbColorImage = cat(3, maskedImage, maskedImage, maskedImage);
        else
            % It's really an RGB image already.
            rgbColorImage = maskedImage;
        end
        
        % Extract the individual red, green, and blue color channels.
        redChannel = maskedImage(:, :, 1);
        greenChannel = maskedImage(:, :, 2);
        blueChannel = maskedImage(:, :, 3);
        
        w=uint8(b);
        x=uint8(c);
        y=uint8(d);
 
        desiredColor = [w, x, y]; % Purple
        % Make the red channel that color
        
        redChannel(edge2) = desiredColor(1);
        greenChannel(edge2) = desiredColor(2);
        blueChannel(edge2) = desiredColor(3);        
        maskedImage = cat(3, redChannel, greenChannel, blueChannel);  

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Foreground = maskedImage;
        Foreground = im2double(Foreground)*350-50;

        % load background image
        Background = im2double(origHair);

        alpha = bsxfun(@times, ones(size(Foreground,1), size(Foreground,2)), .4);

        % find a scale dynamically with some limit
        Foreground_min = min( min(Foreground(:)), -50 );
        Foreground_max = max( max(Foreground(:)), 300 );

        % overlay the image by blending
        Background_blending = bsxfun(@times, Background, bsxfun(@minus,1,alpha));
        Foreground_blending = bsxfun( @times, bsxfun( @rdivide, ...
            bsxfun(@minus, Foreground, Foreground_min), ... 
            Foreground_max-Foreground_min ), alpha );

        out = bsxfun(@plus, Background_blending, Foreground_blending);
        
        imgGaussian=imgaussfilt(out,2); 
                
        finalImage= imadd(im2uint8(imgGaussian), im2uint8(images)); 
    
    imwrite(finalImage,fname,'jpg')
    fname = strcat('out/', a);
    newImage=fname;
end