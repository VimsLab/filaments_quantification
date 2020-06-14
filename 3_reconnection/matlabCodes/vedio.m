new_six= imread('output/19/new_six_6.png');
id = 14
file = strcat('C:\work\Lab\Stromules\Reconnection\FromVims\intersection\output_input_v2_modified_11_14\',num2str(id),'\',num2str(id),'L_final.mat');
L = load(file);
L = L.L_struct;


videopath = strcat('C:\work\Lab\Stromules\Reconnection',num2str(id),'_demo_v1.avi');
video = VideoWriter(videopath); %create the video object
video.FrameRate = 5;
open(video); %open the file for writing

original = strcat('original/', num2str(id), '.png');

I = imread(original);
I = im2double(I);
for i = 1 : numel(L) %where N is the number of images
      display(i)
      if numel(L(i).index) > 20  && (L(i).exist == 1) 
        output = zeros(size(new_six(:,:,1)));
        output(L(i).index) = 1;
        output = imdilate(output, ones(3,3));
        rgbImage = cat(3, output, I,  I);
        for ii = 1: 5
        writeVideo(video,rgbImage); %write the image to file
        end
      end 
end
close(video); %close the file

videopath = strcat('C:\work\Lab\Stromules\Reconnection',num2str(id),'_demo_v2.avi');
video2 = VideoWriter(videopath); %create the video object
video2.FrameRate = 5;
open(video2); %open the file for writing

original = strcat('original/', num2str(id), '.png');

I2 = imread(original);
I2 = im2double(I2);
output2 = zeros(size(new_six(:,:,1)));
for i = 1 : numel(L) %where N is the number of images
      display(i)
      if numel(L(i).index) > 20  && (L(i).exist == 1) 
        tmp_canvas = zeros(size(new_six(:,:,1)));
        tmp_canvas(L(i).index) = 1;
        tmp_canvas = imdilate(tmp_canvas, ones(3,3));
        output2 = tmp_canvas | output2;
        rgbImage = cat(3, output2, I2,  I2);
        for ii = 1: 5
        writeVideo(video2,rgbImage); %write the image to file
        end
      end 
end
close(video2); %close the file