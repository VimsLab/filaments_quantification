
function TipSearch_v2(name,outputfoder, inputfolder)

%outputfolder = strcat('output/',num2str(num),'/');

outputfolder = strcat(outputfoder,name,'/');
mkdir(outputfolder);
%input_six = 'test_on_mt_di5/';
display('running, first Pass');
input_six = inputfolder;

im1 = imread(strcat(input_six,name,'_branch_1.png'));
im2 = imread(strcat(input_six,name,'_branch_2.png'));
im3 = imread(strcat(input_six,name,'_branch_3.png'));
im4 = imread(strcat(input_six,name,'_branch_4.png'));
im5 = imread(strcat(input_six,name,'_branch_5.png'));
im6 = imread(strcat(input_six,name,'_branch_6.png'));
im7 = imread(strcat(input_six,name,'_branches_merge.png'));

stack = cat(3, im1, im2, im3, im4, im5, im6, im7);

stack = (im2double(stack)> 0.5);
%% do some cleaning here.  Q: should I do the same thing to the 7th layer?
  %clearn some little gaps and get rid of little fragments
for i = 1 : 7 
     stack(:,:, i) = bwmorph(stack(:,:,i),'bridge');   % bridge will infulence the result.
   % stack(:,:, i) = bwmorph(stack(:,:,i),'bridge');                                                   
     stack(:,:, i) = bwareaopen(stack(:,:,i),5);
end

%stack = stack((250:450),(1:150),:);  %for debug.

%% Find overlaping area
overLap = zeros(size(stack(:,:,1)));
%dilate_for_overlap = imdilate(stack, ones(5,5));
for i = 1 : 6
    for ii = i + 1 : 6
    overLap = ((stack(:,:,i) & stack(:,:,ii)) | overLap);
    %overLap = ((dilate_for_overlap(:,:,i) & dilate_for_overlap(:,:,ii)) | overLap);
    end
end
% overLap = bwareaopen(overLap, 5);
% for i = 1 : 6
%     %stack(:,:,i) = stack(:,:,i) & (~overLap);  
% end
% 
 %imshow(stack(:,:,1), [])
% imshow(overLap, [])




%% assign label
L = zeros(size(stack));
Ln = zeros(1,7);
%% BeforeMerge Write
for i = 1 : 6
    [L(:,:,i), Ln(i)] = bwlabel(stack(:,:,i));   
    %imwrite(label2rgb(L(:, :,i),'hsv', 'k', 'shuffle'), strcat(outputfolder,'before_merge',num2str(i),'.png')) ;
end
%imwrite(label2rgb(overLap,'hsv', 'k', 'shuffle'),'checkoverlap.png')
%%
%% Merge heavily overlapped, if a fragments' 40% length belongs to the other one length. merge to the longer one
% Find what are they.
[Lo, Lon] = bwlabel(overLap);

msg = 'Processed Rate: 0.0%';

nmsg = numel(msg)+1;

for i = 1 : Lon %label of overlap
    
    rate = i/Lon;
    fprintf(repmat('\b',1,nmsg)) %deletethis line
    msg = sprintf('%s%0.2f%%','Processed Rate: ',rate);
    disp(msg);
    nmsg = numel(msg)+1;
    pause(0.01)

    klo = find(Lo == i); %find overlap label location
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    For debug, visualizing
%     for_check = overLap;
%     for_check = double(for_check);
%     for_check(klo) = 2;
%     imshow(label2rgb(for_check,'hsv','k', 'shuffle'),[]);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%     componets = initializeComponents();

    record = zeros(4,10);
    n_label = 1; 
    
    for ii = 1 : 6
        tmp = L(:,:,ii);  %tmp  
        label = tmp(klo);  %what the label is at there. what if two are there??
        %display(label(1)); 
        label = unique(label);
        for iii = 1 : numel(label)
            if(label(iii) ~= 0)
                record(1, n_label) = label(iii);
                record(2, n_label) = numel(find(tmp == record(1,n_label))); % how many are there 
                record(3, n_label) = numel(klo)/record(2,n_label);
% %                 if (numel(klo) / record(2, n_label) >  1)
% % 
% %                 end
                record(4,n_label) = ii; 
                n_label = n_label + 1;
    %             [x1, y1, x2, y2] = find_tip(label(1), tmp);
    %         
    %             record(5,ii) = x1;
    %             record(6,ii) = y1;
    %             record(7,ii) = x2;
    %             record(8,ii) = y2;
                %display(record);
            end
        end
    end
    %display(record)
    valid_record_ind = record(1,:) > 0; % get valid record where label is not 0
    record = record(:,valid_record_ind);
    [Min, Imin] = min(record(3,:));  
    k = 3 ;
   
    convert_to = record(1,Imin);
    to_be_converted = find_to_be_converted(Imin,record,L, k);
    %display( to_be_converted )
    for ii = 1 : numel(to_be_converted)  
        %if record(1,to_be_converted(ii)) ~= convert_to
           layer_location = record(4, to_be_converted(ii));
           label_num = record(1, to_be_converted(ii));
          % k_to_be_convert = find(L(:,:,layer_location) == label_num); % find all the label, to be convert.
           tmp_3 = L(:,:,layer_location );
           k_to_be_convert = find(tmp_3 == label_num);
           
           tmp_2 = L(:,:, record(4,Imin)); %record(4,Imin) the future layer of the label
           tmp_2(k_to_be_convert) = convert_to;
           
           tmp_3(k_to_be_convert) = 0;


           L(:,:, record(4,Imin)) = tmp_2; 
           L(:,:,layer_location) = tmp_3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            if record(4,Imin) == layer_location
%                display(100000000000000000000000000000000000000000000000000000000000)
%            end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %end
    end

     
    
end

%% Save after first pass, namely overlapping case
% 
% Lmat = strcat(outputfolder,name,'_L.mat');
% save(Lmat, 'L');
% 
% for i = 1 : 6
%     imwrite(label2rgb(L(:, :,i),'hsv', 'k', 'shuffle'), strcat(outputfolder,name,'_After_merge_check_',num2str(i),'.png')) ;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% post process  

    old_six = L;

    new_six = zeros(size(old_six(:,:,6)));
    map_for_connect = zeros(size(new_six));
    
    for i = 1 : 6
        map_for_connect((old_six(:,:,i)> 0)) = 1;
    end
    
%     Note!!!!!!!!!!! l is l 1 is 1!!!! this is my bad, change later
% or now? I need to comeup with a new name first
    
    label_num = 1;
    for i = 1 : 6
        tmp_L = old_six(:,:,i);
        update_L = zeros(size(tmp_L));
        
        label_l = unique(tmp_L);
        label_l = label_l(label_l > 0);
        for ii = 1 : numel(label_l)
            canvas = zeros(size(tmp_L));
            pixels = find(tmp_L == label_l(ii));
            canvas(pixels) = 1;
            canvas = bwareaopen(canvas,4);
            canvas = bwmorph(canvas,'skel', Inf);
            
            
            %I'll leave this case for the future
            [label_canvas,n_canvas] = bwlabel(canvas);
           % imshow(label2rgb(label_canvas,'hsv', 'k', 'shuffle'),[])
            if n_canvas > 1
                
%                 figure(1);
%                 imshow(label2rgb(label_canvas,'hsv', 'k', 'shuffle'),[]);
              
                all_tips = bwmorph(canvas,'endpoints')>0; %find all endpoints
                canvas_2 = zeros(size(tmp_L));
                canvas_2(all_tips) = 1;                               
                for ii = 1 : n_canvas - 1
                     [lt1, rt1] = find_tip(ii, label_canvas);
                     [lt2, rt2] = find_tip(ii + 1, label_canvas);
%                      Dlt = bwdistgeodesic(map_for_connect, lt(2), lt(1));
%                      for iii = 1:numel(all_tips)
                     index = draw_line(rt1, lt2,size(canvas)); 
                     canvas(index) = 1;

                     
%                     end
                    
                end
%                 figure (2)
%                 imshow(label2rgb(canvas,'hsv', 'k', 'shuffle'),[]);

            end
            canvas_dialte = imdilate(canvas, ones(3));  %for future reconnect
            if sum(canvas,'all') ~= 0 %after skel, and areaopen, some might be 0
                L_struct(label_num).index = find(canvas>0);
                L_struct(label_num).layer = i;
                L_struct(label_num).ind_dilate = find(canvas_dialte > 0);
                [L_struct(label_num).lt, L_struct(label_num).rt] = find_tip(1,canvas);
                L_struct(label_num).exist = 1;
                [L_struct(label_num).ld, L_struct(label_num).rd] = ... 
                        get_dir(canvas, L_struct(label_num).lt, L_struct(label_num).rt, 10);
               %NOTICE! MY BAD. ITS FROM TIP POINT TO INSIDE OF THE
               %FRAGMENTS!!!
                update_L(canvas > 0) = label_num;
                label_num = label_num + 1;
                %imshow(label2rgb(update_L,'hsv', 'k', 'shuffle'),[]);
            end
            
        end
        new_six(:,:,i) = update_L;
        
    end
%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     new_six_mat = strcat(outputfolder,name,'new_six.mat');
%     struct_mat = strcat(outputfolder,name,'L_struct.mat');
%     save(new_six_mat, 'new_six');
%     save(struct_mat, 'L_struct');
 %%   
% end
%original = strcat('original/', name, '.png');


msg = 'Processed Rate: 0.0%';

nmsg = numel(msg)+1;


update_state = 1;

for i = 1 : numel(L_struct)
    
    rate = i / numel(L_struct);
    fprintf(repmat('\b',1,nmsg)) %deletethis line
    msg = sprintf('%s%0.2f%%','Reconnecting Rate: ',rate);
    disp(msg);
    nmsg = numel(msg)+1;
    pause(0.01)
    
%     map_for_check = zeros(size(new_six(:,:,1)));
%     map_for_check(L_struct(i).index) = 1;
%     map_for_checkB = map_for_check;
    %subplot(2,1,1);
    %imshow(label2rgb(map_for_check,'hsv', 'k', 'shuffle'),[]);
    update_state = 1;
    while(update_state == 1)
%         if(L_struct(392).exist == 0)
%         end

        update_state = 0;
        label = i;
        search_size = 50;
        [update_state, L_struct] = reconnect(label, new_six, L_struct, search_size, update_state);
        
       % subplot(2,1,2);

        %imshow(label2rgb(map_for_check,'hsv', 'k', 'shuffle'),[]);    

    end
    %close all
end

output = zeros(size(new_six(:,:,1)));
output_dilate = zeros(size(new_six(:,:,1)));
for i = 1 : numel(L_struct)
    if L_struct(i).exist == 1
        output(L_struct(i).index) = i;
        output_dilate(L_struct(i).ind_dilate) = i;
    end
    
end

    imwrite(label2rgb(output,'hsv', 'k', 'shuffle'),strcat(outputfolder,name,'output.png'));
    imwrite(label2rgb(output_dilate,'hsv', 'k', 'shuffle'),strcat(outputfolder,name,'output_dilate.png'));
% for i = 1 : 6
%     imwrite(label2rgb(new_six(:, :,i),'hsv', 'k', 'shuffle'), strcat(outputfolder,'new_six_',num2str(i),'.png')) ;
% end


    new_six_mat = strcat(outputfolder,name,'final_six.mat');
    struct_mat = strcat(outputfolder,name,'L_final.mat');
    save(new_six_mat, 'new_six');
    save(struct_mat, 'L_struct');
    
%% For each instance quantifying anlysis
six6 = new_six;
L = L_struct;
display('quantifying analysis');
number = 1;
sumCurv = 0;
totalLen = 0;
geodestic = true;
azimuthAngle.x = [];
azimuthAngle.y = [];
azimuthAngle.angle = [];
canvas_azimuthAngle = zeros(size(six6(:,:,1)));

index_num = [];
length_all = []; %length for each MT
aveCurvature_all = [];
aveAzimuthAngle_all = [];
curvature_all = [];
azimuthAngle_all = [];

filename_1 = 'filaments_info.xlsx';
filename_2 = 'summary_info.xlsx';
filename_3 = 'curvature.xlsx';
filename_4 = 'azimuthAngle.xlsx';

for i = 1 : numel(L)
    if numel(L(i).index) > 20  && (L(i).exist == 1)
        
   
   % canvas = zeros(size(six6(:,:,1)));
    [vertices_y, vertices_x] = ind2sub(size(six6(:,:,1)), L(i).index);
    
    %% Convert Index sequence from start of the filament to end of the filament.. 
        if geodestic == true
            
            canvas = zeros(size(six6(:,:,1)));
            canvas(L(i).index) = 1;
            %Following is to make sure find the left most end point. 
            %because sometimes left most point can not be detected as end
            %point: * *
            %       * *
            %          *
            %           **** 
            %              **********         
            %%%%%%%%%%%%%%%%%%%%%%%%%%% like above. 
            canvas = imdilate(canvas, ones(3,3));
            canvas = bwmorph(canvas, 'skel', Inf);
            [row_y, col_x]  = find( bwmorph(canvas, 'endpoints'));
            num_of_endpoint = size(row_y);       
            [x1,ind] = min(col_x);
            y1 = row_y(ind);
            left = [y1, x1];
            D = bwdistgeodesic(logical(canvas),left(2), left(1)); 
            D = reshape(D, [], 1);
            [sortD, sortIndex] = sort(D);
            start = find(sortD > 0);
            newindex = sortIndex(start);
            
        end 
    [vertices_y, vertices_x] = ind2sub(size(six6(:,:,1)), newindex);
    
    sx = polyfit( 1:size(newindex,1), (vertices_x)', 4);
    
    sy = polyfit(1:size(newindex,1), (vertices_y)', 4);
    
    
    fx = polyval(sx, 1:size(newindex,1));
    fy = polyval(sy, 1:size(newindex,1));
        
    %% azimuth Angle:
    length = size(newindex,1);

    if length > 10
        
        azimuthAngle_current = [];
        %%% Use pixel instead of using poly func
        %fy = vertices_y;
        %fx = vertices_x;
%         fy = fy';
%         fx = fx';
        step = 1;
        for eachSeg = 1 : size(newindex,1) - step
            %yDiff = fy(eachSeg + 1) - fy(eachSeg)  + 0.000000000001;
            % xDiff = fx(eachSeg + 1) - fx(eachSeg);
           
            yDiff = fy(eachSeg + step) - fy(eachSeg)  + 0.000000000001;
            xDiff = fx(eachSeg + step) - fx(eachSeg);
            azimuthAngle.x = [azimuthAngle.x, fx(eachSeg)] ;
            azimuthAngle.y = [azimuthAngle.y, fy(eachSeg)];
            azimuthAngle.angle = [azimuthAngle.angle, (atan(yDiff/xDiff + 0.000000001)) + 3.1416926/2];
            
            azimuthAngle_current = [azimuthAngle_current, (atan(yDiff/xDiff + 0.000000001)) + 3.1416926/2];            
            
            if (floor(fx(eachSeg))>0 && floor(fy(eachSeg)) > 0 && floor(fy(eachSeg)) <= size(six6(:,:,1),2) && floor(fx(eachSeg)) <= size(six6(:,:,1),1)        )
                canvas_azimuthAngle( floor( fy(eachSeg)),floor( fx(eachSeg))) = (atan(yDiff/xDiff + 0.000000001) + 3.1416926/2) / 3.1415926 * 2;
            end
        end
        aveAzimuthAngle_all = [aveAzimuthAngle_all; nansum(azimuthAngle_current, 'all')/length];
        
    else
        azimuthAngle.x = fx(1);
        azimuthAngle.y = fy(1);
        azimuthAngle.angle = [azimuthAngle.angle, (atan(fy(end) - fy(1)/ (fx(end) - fx(1) + 0.000000001)) + 3.1416926/2 )];
        
        aveAzimuthAngle_all = [aveAzimuthAngle_all; nansum((atan(fy(end) - fy(1)/ (fx(end) - fx(1) + 0.000000001))+ 3.1416926/2 ), 'all')/length];
        
        canvas_azimuthAngle(floor(fy(1)), fx(1)) = (atan(fy(end) - fy(1)/ fx(end) - fx(1) + 0.000000001)+ 3.1416926/2) / 3.1415926 * 2;
    % 180 is the same as 0 degree. that is why I divide by 2. Also PI is
    % because I want to normalize to [0,1]
    end
    
    %% curvature
    p = polyfit(vertices_x, vertices_y, 4);   
    
    dx = polyder(p);
    ddx = polyder(dx);
    curvature = polyval(ddx, vertices_x)./ (1 + polyval(dx, vertices_x).^2).^1.5;

    L(i).curvature = curvature;
    L(i).aveCurvature = sum(curvature)/size(curvature, 1);

   % Vertices = [vertices_x, vertices_y];
   % N=LineNormals2D(Vertices);
   % k=LineCurvature2D(Vertices);
   % k=k*100;
   % plot([Vertices(:,1) Vertices(:,1)+k.*N(:,1)]',[Vertices(:,2) Vertices(:,2)+k.*N(:,2)]','g');
    %plot([Vertices(Lines(:,1),1) Vertices(Lines(:,2),1)]',[Vertices(Lines(:,1),2) Vertices(Lines(:,2),2)]','b');
  %  plot(Vertices(:,1),Vertices(:,2),'r.');
     
     totalLen = length + totalLen;
     sumCurv = L(i).aveCurvature + sumCurv;
     
     index_num = [index_num; number];
     length_all = [length_all; length];
     aveCurvature_all = [aveCurvature_all; L(i).aveCurvature];
     curvature_all = [curvature_all; curvature];
     number = 1 + number
     

    end  
    
end
canvas_azimuthAngle(isnan(canvas_azimuthAngle)) = 0;
azimuthAngle_map = imdilate (canvas_azimuthAngle, ones(3,3));
mask_white = azimuthAngle_map;
mask_white(mask_white>0) = 1;
rgb = ind2rgb(gray2ind(azimuthAngle_map,255),jet(255));
rgb = rgb.*mask_white;
imwrite(rgb,strcat(outputfolder,name,'AzimuthAngle_map.png'));


table_1 = table(index_num, length_all, aveCurvature_all, aveAzimuthAngle_all);
filaments_data = strcat(outputfolder,name,'filaments_data.xlsx');
writetable(table_1,filaments_data,'Sheet',1,'Range','A1')

[numOfFilament,~] = size(index_num);
aveLength = sum(length_all)/numOfFilament;
aveCurvature = sum(aveCurvature_all,'all')/ numOfFilament;
aveAzimuthAngle = sum((aveAzimuthAngle_all))/numOfFilament;


table_2 = table(numOfFilament, aveLength, aveCurvature, aveAzimuthAngle);
 
summary_data = strcat(outputfolder,name,'filaments_data.xlsx');
writetable(table_2,summary_data,'Sheet',2,'Range','A1');

table_3 = table(curvature_all);
curvature_data = strcat(outputfolder,name,'filaments_data.xlsx');
writetable(table_3,curvature_data,'Sheet',3,'Range','A1');

azimuthOut = reshape(azimuthAngle.angle, [],1);
table_4 = table(azimuthOut);
azimuthAngle_data = strcat(outputfolder,name,'filaments_data.xlsx');
writetable(table_4,azimuthAngle_data,'Sheet',4,'Range','A1');    
    
    
    
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % I2 = imread(original);
% % % 
% % % 
% % % videopath = strcat(outputfolder,num2str(num),'_demo_v1.avi');
% % % video = VideoWriter(videopath); %create the video object
% % % video.FrameRate = 2;
% % % open(video); %open the file for writing
% % % 
% % % original = strcat('original/', num2str(num), '.png');
% % % 
% % % I = imread(original);
% % % I = im2double(I);
% % % for i = 1 : numel(L_struct) %where N is the number of images
% % %       display(i)
% % %       if L_struct(i).exist == 1
% % %         output = zeros(size(new_six(:,:,1)));
% % %         output(L_struct(i).index) = 1;
% % %         output = imdilate(output, ones(3,3));
% % %         rgbImage = cat(3, output, I,  I);
% % %         writeVideo(video,rgbImage); %write the image to file
% % %       end 
% % % end
% % % close(video); %close the file
% % % 
% % % videopath = strcat(outputfolder,num2str(num),'_demo_v2.avi');
% % % video2 = VideoWriter(videopath); %create the video object
% % % video2.FrameRate = 2;
% % % open(video2); %open the file for writing
% % % 
% % % original = strcat('original/', num2str(num), '.png');
% % % 
% % % I2 = imread(original);
% % % I2 = im2double(I2);
% % % output2 = zeros(size(new_six(:,:,1)));
% % % for i = 1 : numel(L_struct) %where N is the number of images
% % %       display(i)
% % %       if L_struct(i).exist == 1
% % %         tmp_canvas = zeros(size(new_six(:,:,1)));
% % %         tmp_canvas(L_struct(i).index) = 1;
% % %         tmp_canvas = imdilate(tmp_canvas, ones(3,3));
% % %         output2 = tmp_canvas | output2;
% % %         rgbImage = cat(3, output2, I2,  I2);
% % %         writeVideo(video2,rgbImage); %write the image to file
% % %       end 
% % % end
% % % close(video2); %close the file
% function component = group(label, layer, tmp_L,L)
%     canvas = zeros(size(tmp_1));
%     for ii = 1 : numel(label_l)
%        [pixels, num] = find(tmp_L == label_l);
%        canvas(pixels) = 1;
%        canvas = bwareaopen(canvas,4);
%        searching_area = imdialte(canvas, ones(3,3))
%     end
% 
% 
% end
% 
% % 
% for i = 1 : 6
%     tmp_L = L(:,:, i);
%     label_l = unique(tmp_L);
%     for ii = 1 : numel(label_l)
%         
%     end
% end
