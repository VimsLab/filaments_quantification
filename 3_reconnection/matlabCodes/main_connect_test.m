

%new_six = load('C:\work\Lab\Stromules\Reconnection\FromVims\intersection\output\0\0new_six.mat');
new_six = load('C:\work\Lab\Stromules\Reconnection\FromVims\intersection\output_input\0\0new_six.mat');

L_struct= load('C:\work\Lab\Stromules\Reconnection\FromVims\intersection\output_test_on_mt_di5_v2\3\3_L.mat');
L_struct = L_struct.L_struct;
new_six = new_six.new_six;
tic
num = 0;
                
                
%original = strcat('original/', num2str(num), '.png');
original = '0_org.png' ;
I2 = imread(original);

update_state = 1;
for i = 1 : numel(L_struct)

    map_for_check = zeros(size(new_six(:,:,1)));
    map_for_check(L_struct(i).index) = 1;
    map_for_check = imdilate(map_for_check, ones(3,3));
    map_for_checkB = map_for_check;
    %subplot(2,1,1);
    %imshow(label2rgb(map_for_check,'hsv', 'k', 'shuffle'),[]);
    update_state = 1;
    while(update_state == 1)
%         if(L_struct(392).exist == 0)
%         end
          if i == 3 || i == 76
          end
        update_state = 0;
        label = i;
        search_size = 15;
        [update_state, L_struct] = reconnect_onelayer(label, new_six, L_struct, search_size, update_state, L_struct(i).layer);
        
       % subplot(2,1,2);
        map_for_check = zeros(size(new_six(:,:,1)));
        map_for_check(L_struct(i).index) = 1; 
        map_for_check = imdilate(map_for_check, ones(5,5));
        %imshow(label2rgb(map_for_check,'hsv', 'k', 'shuffle'),[]);    
        if update_state == 1
%             rgbcheck = cat(3, map_for_check * 255, 255 * map_for_checkB,   I2/ I2(:,:,1));
            I2_temp = double(I2)/max(double(I2(:)));
            rgbcheck =  cat(3, map_for_check * 255, 255 * map_for_checkB, I2_temp * 255);
            imshow(rgbcheck, []);
           imshowpair(map_for_check,map_for_checkB,'blend','Scaling','joint')
         pause(2);
        else
            %rgbcheck = cat(3, map_for_check, I2,  I2);
            I2_temp = double(I2)/max(double(I2(:)));
            rgbcheck =  cat(3, map_for_check * 255, 255 * map_for_check, I2_temp * 255);
            imshow(rgbcheck, []);
         %  pause(2);
        end
    end
    %close all
end

%%%%%%%%%%%%%%%%%%%%%
update_state = 1;
for i = 1 : numel(L_struct)
    
    map_for_check = zeros(size(new_six(:,:,1)));
    map_for_check(L_struct(i).index) = 1;
    map_for_check = imdilate(map_for_check, ones(3,3));
    map_for_checkB = map_for_check;
    %subplot(2,1,1);
    %imshow(label2rgb(map_for_check,'hsv', 'k', 'shuffle'),[]);
    update_state = 1;
    while(update_state == 1)
%         if(L_struct(392).exist == 0)
%         end
          if i == 3 || i == 76
          end
        update_state = 0;
        label = i;
        search_size = 50;
        [update_state, L_struct] = reconnect(label, new_six, L_struct, search_size, update_state);
        
       % subplot(2,1,2);
        map_for_check = zeros(size(new_six(:,:,1)));
        map_for_check(L_struct(i).index) = 1; 
        map_for_check = imdilate(map_for_check, ones(5,5));
        %imshow(label2rgb(map_for_check,'hsv', 'k', 'shuffle'),[]);    
        if update_state == 1
%             rgbcheck = cat(3, map_for_check * 255, 255 * map_for_checkB,   I2/ I2(:,:,1));
            I2_temp = double(I2)/max(double(I2(:)));
            rgbcheck =  cat(3, map_for_check * 255, 255 * map_for_checkB, I2_temp * 255);
            imshow(rgbcheck, []);
         %  imshowpair(map_for_check,map_for_checkB,'blend','Scaling','joint')
        pause(2);
        else
            %rgbcheck = cat(3, map_for_check, I2,  I2);
            I2_temp = double(I2)/max(double(I2(:)));
            rgbcheck =  cat(3, map_for_check * 255, 255 * map_for_check, I2_temp * 255);
            imshow(rgbcheck, []);
           pause(2);
        end
    end
    %close all
end

output = zeros(size(new_six(:,:,1)));
for i = 1 : numel(L_struct)
    if L_struct(i).exist == 1
       output(L_struct(i).index) = i;
    end
    
end

    imwrite(label2rgb(output,'hsv', 'k', 'shuffle'),strcat(outputfolder,num2str(num),'output.png'));

for i = 1 : 6
    imwrite(label2rgb(new_six(:, :,i),'hsv', 'k', 'shuffle'), strcat(outputfolder,'new_six_',num2str(i),'.png')) ;
end

toc


videopath = strcat(outputfolder,num2str(num),'_demo_v1.avi');
video = VideoWriter(videopath); %create the video object
video.FrameRate = 2;
open(video); %open the file for writing

original = strcat('original/', num2str(num), '.png');

I = imread(original);
I = im2double(I);
for i = 1 : numel(L_struct) %where N is the number of images
      display(i)
      if L_struct(i).exist == 1
        output = zeros(size(new_six(:,:,1)));
        output(L_struct(i).index) = 1;
        output = imdilate(output, ones(3,3));
        rgbImage = cat(3, output, I,  I);
        writeVideo(video,rgbImage); %write the image to file
      end 
end
close(video); %close the file

videopath = strcat(outputfolder,num2str(num),'_demo_v2.avi');
video2 = VideoWriter(videopath); %create the video object
video2.FrameRate = 15;
open(video2); %open the file for writing

original = strcat('original/', num2str(num), '.png');

I2 = imread(original);
I2 = im2double(I2);
output2 = zeros(size(new_six(:,:,1)));
for i = 1 : numel(L_struct) %where N is the number of images
      display(i)
      if L_struct(i).exist == 1
        tmp_canvas = zeros(size(new_six(:,:,1)));
        tmp_canvas(L_struct(i).index) = 1;
        tmp_canvas = imdilate(tmp_canvas, ones(3,3));
        output2 = tmp_canvas | output2;
        rgbImage = cat(3, output2, I2,  I2);
        writeVideo(video2,rgbImage); %write the image to file
      end 
end
close(video2); %close the file
toc