
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
seg = imread(strcat(input_six,name,'.png'));

seg = im2double(seg)>0.5;

stack = cat(3, im1, im2, im3, im4, im5, im6, im7);

stack = (im2double(stack)> 0.5);
%% do some cleaning here.  Q: should I do the same thing to the 7th layer?
  %clearn some little gaps and get rid of little fragments
for i = 1 : 7 
     stack(:,:, i) = bwmorph(stack(:,:,i),'bridge');   % bridge will infulence the result.                                                 
     stack(:,:, i) = bwareaopen(stack(:,:,i),5);
end
%% Find overlaping area
%% assign label
%% BeforeMerge Write
%% Merge heavily overlapped, if a fragments' 40% length belongs to the other one length. merge to the longer one


% %% Save after first pass, namely overlapping case
% % 

% Lmat = strcat(outputfolder,name,'_L.mat');
% % save(Lmat, 'L');
% 
%% post process  
    old_six = stack(:,:,1:6);
    new_six = zeros(size(old_six(:,:,6)));
    map_for_connect = zeros(size(new_six));
    
    for i = 1 : 6
        map_for_connect((old_six(:,:,i)> 0)) = 1;
    end
    
    label_num = 1;
    for i = 1 : 6
        tmp_L = old_six(:,:,i);
        update_L = zeros(size(tmp_L));
        [label_canvas,n_canvas] = bwlabel(tmp_L);
        label_l = unique(label_canvas);
        label_l = label_l(label_l > 0);
        for ii = 1 : n_canvas
            ii
            canvas = label_canvas == label_l(ii);

            canvas = bwmorph(canvas,'skel', Inf);
            
            canvas_dialte = imdilate(canvas, ones(5));  %for future reconnect
            if sum(canvas(:)) ~= 0 %after skel, and areaopen, some might be 0
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
    new_six_mat = strcat(outputfolder,name,'new_six.mat');
    struct_mat = strcat(outputfolder,name,'L_struct.mat');
%     
% %     new_six_mat = strcat(outputfolder,name,'final_six.mat');
% %     struct_mat = strcat(outputfolder,name,'L_final.mat');
    save(new_six_mat, 'new_six');
    save(struct_mat, 'L_struct');
 %%   
% end
%original = strcat('original/', name, '.png');
% new_six = load(new_six_mat);
% L_struct = load(struct_mat);
% L_struct = L_struct.L_struct;
% new_six = new_six.new_six;

% 
% L_struct_new(12)= L_struct(12);
% L_struct_new(15)= L_struct(15);
% L_struct_new(504)= L_struct(504);
% L_struct_new(506)= L_struct(506);
% 
% L_struct = L_struct_new;
% 
% ind_for_debug_504 = find(new_six == 504);
% ind_for_debug_506 = find(new_six == 506);
% ind_for_debug_12 = find(new_six == 12);
% ind_for_debug_15 = find(new_six == 15);
% 
% q =[12;15; 504; 506]; 
% 
% new_six = zeros(size(new_six));
% 
% new_six(ind_for_debug_506) =506;
% new_six(ind_for_debug_504) = 504;
% new_six(ind_for_debug_12) = 12;
% new_six(ind_for_debug_15) = 15;


msg = 'Processed Rate: 0.0%';

nmsg = numel(msg)+1;



update_state = 1;
for i = 1 : numel(L_struct)
      
     rate = i / numel(L_struct);
%      fprintf(repmat('\b',1,nmsg)) %deletethis line
     msg = sprintf('%s%0.2f%%','Reconnecting Rate: ',rate);
     disp(msg);
%     nmsg = numel(msg)+1;
%     pause(0.01)
    
%     map_for_check = stack(:,:, 7);
%     map_for_check(L_struct(i).index) = 244;
%     map_for_checkB = map_for_check;

    update_state = 1;
    while(update_state == 1)
        update_state = 0;
        label = i;
        search_size = 50;
        [update_state, L_struct, new_six] = reconnect_v2(label, new_six, L_struct, search_size, update_state, seg);
        
%        if update_state == 1
%         map_for_check(L_struct(i).ind_dilate) = 123;
%         imshow(label2rgb(map_for_check,'hsv', 'k', 'shuffle'),[]);    
%        end
    end
end

output = zeros(size(new_six(:,:,1)));
output_dilate = zeros(size(new_six(:,:,1)));
for i = 1: numel(L_struct)
   
    
    
    if L_struct(i).exist == 1
        output(L_struct(i).index) = i;
        output_dilate(L_struct(i).ind_dilate) = i;
    end
    
end

    imwrite(label2rgb(output,'hsv', 'k', 'shuffle'),strcat(outputfolder,name,'_output.png'));
    imwrite(label2rgb(output_dilate,'hsv', 'k', 'shuffle'),strcat(outputfolder,name,'_output_dilate.png'));

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
    
    
