% % six1 = imread('output/0/new_six_1.png');
% % six2 = imread('output/0/new_six_2.png');
% % six3 = imread('output/0/new_six_3.png');
% % six4 = imread('output/0/new_six_4.png');
% % six5 = imread('output/0/new_six_5.png');
%six6 = imread('output/19/new_six_6.png');
six6 = imread('C:\work\Lab\Stromules\Reconnection\FromVims\intersection\Kody_data\output\0\new_six_6.png');
%L = load('output/0/0L_struct.mat');
%???????
%L = load('C:\work\Lab\Stromules\Reconnection\FromVims\intersection\output_input_v2_modified_11_14\3\3L_final.mat');
%L = load('C:\work\Lab\CVPR2019\PlanktothrixDemoSet\final\400\400L_final.mat');

id = 1;
%file = strcat('C:\work\Lab\CVPR2019\ROAD\Final_v2_bt_distance10\',num2str(id),'\',num2str(id),'L_final.mat');
file = strcat('C:\work\Lab\Stromules\Reconnection\FromVims\intersection\reconnectionCode\Kody_data\output\',num2str(id),'\',num2str(id),'L_final.mat');
L = load(file);
L = L.L_struct;
canvas = zeros(size(six6(:,:,1)));
output = zeros(size(six6(:,:,1)));


canvas = zeros(2044);
output = zeros(2044);



number = 1;
for i = 1 : numel(L)
    if numel(L(i).index) > 20  && (L(i).exist == 1) 
   % canvas = zeros(size(six6(:,:,1)));
    
    canvas = zeros(size(canvas));
    
    canvas(L(i).index) = 1;
    canvas = imdilate(canvas, ones(3));
    new_ind = find(canvas);
    output(new_ind) = i;
    number = 1 + number
    end    
end

  % imwrite(label2rgb(output,'hsv', 'k', 'shuffle'), strcat('0_for_manual/400', num2str(number),'final.png'));
%imwrite(label2rgb(output,'hsv', 'k', 'shuffle'), strcat('C:\work\Lab\CVPR2019\ROAD\Final\3\', num2str(number),'final.png'));
%imwrite(label2rgb(output,'hsv', 'k', 'shuffle'), strcat('C:\work\Lab\CVPR2019\ROAD\road',num2str(id), num2str(number),'final.png'));
imwrite(label2rgb(output,'hsv', 'k', 'shuffle'), strcat('C:\work\Lab\Stromules\Reconnection\FromVims\intersection\Kody_data\output\',num2str(id), num2str(number),'final.png'));
% % 
% % function forManual(num)
% % 
% % outputfolder = strcat('output/',num2str(num),'/');
% % mkdir(outputfolder);
% % input_six = 'test_on_mt_di5/';
% % 
% % im1 = imread(strcat(input_six,num2str(num),'_1_predicted.png'));
% % im2 = imread(strcat(input_six,num2str(num),'_2_predicted.png'));
% % im3 = imread(strcat(input_six,num2str(num),'_3_predicted.png'));
% % im4 = imread(strcat(input_six,num2str(num),'_4_predicted.png'));
% % im5 = imread(strcat(input_six,num2str(num),'_5_predicted.png'));
% % im6 = imread(strcat(input_six,num2str(num),'_6_predicted.png'));
% % im7 = imread(strcat(input_six,num2str(num),'_7_predicted.png'));
% % 
% % stack = cat(3, im1, im2, im3, im4, im5, im6, im7);
% % 
% % stack = (im2double(stack)> 0.5);
% % %% do some cleaning here.  Q: should I do the same thing to the 7th layer?
% %   %clearn some little gaps and get rid of little fragments
% % for i = 1 : 7 
% %      %stack(:,:, i) = bwmorph(stack(:,:,i),'bridge');   % bridge will infulence the result.
% %                                                        
% %      stack(:,:, i) = bwareaopen(stack(:,:,i),10);
% % end
% % 
% % %stack = stack((250:450),(1:150),:);  %for debug.
% % 
% % %% Find overlaping area
% % overLap = zeros(size(stack(:,:,1)));
% % %dilate_for_overlap = imdilate(stack, ones(5,5));
% % for i = 1 : 6
% %     for ii = i + 1 : 6
% %     overLap = ((stack(:,:,i) & stack(:,:,ii)) | overLap);
% %     %overLap = ((dilate_for_overlap(:,:,i) & dilate_for_overlap(:,:,ii)) | overLap);
% %     end
% % end
% % % overLap = bwareaopen(overLap, 5);
% % % for i = 1 : 6
% % %     %stack(:,:,i) = stack(:,:,i) & (~overLap);  
% % % end
% % % 
% % % imshow(stack(:,:,1), [])
% % % imshow(overLap, [])
% % 
% % 
% % 
% % 
% % %% assign label
% % L = zeros(size(stack));
% % Ln = zeros(1,7);
% % for i = 1 : 6
% %     [L(:,:,i), Ln(i)] = bwlabel(stack(:,:,i));   
% %     imwrite(label2rgb(L(:, :,i),'hsv', 'k', 'shuffle'), strcat(outputfolder,'before_merge',num2str(i),'.png')) ;
% % end
% % %imwrite(label2rgb(overLap,'hsv', 'k', 'shuffle'),'checkoverlap.png')
% % 
% % %% Merge heavily overlapped, if a fragments' 40% length belongs to the other one length. merge to the longer one
% % % Find what are they.
% % [Lo, Lon] = bwlabel(overLap);
% % 
% % 
% % 
% % 
% % 
% % for i = 1 : Lon %label of overlap
% %     display(Lon);
% %     display(i);
% %     klo = find(Lo == i); %find overlap label location
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    For debug, visualizing
% % %     for_check = overLap;
% % %     for_check = double(for_check);
% % %     for_check(klo) = 2;
% % %     imshow(label2rgb(for_check,'hsv','k', 'shuffle'),[]);
% %  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
% % %     componets = initializeComponents();
% %     record = zeros(4,10);
% %     n_label = 1; 
% %     
% %     for ii = 1 : 6
% %         tmp = L(:,:,ii);  %tmp  
% %         label = tmp(klo);  %what the label is at there. what if two are there??
% %         %display(label(1)); 
% %         label = unique(label);
% %         for iii = 1 : numel(label)
% %             if(label(iii) ~= 0)
% %                 record(1, n_label) = label(iii);
% %                 record(2, n_label) = numel(find(tmp == record(1,n_label))); % how many are there 
% %                 record(3, n_label) = numel(klo)/record(2,n_label);
% %                 if (numel(klo) / record(2, n_label) >  1)
% % 
% %                 end
% %                 record(4,n_label) = ii; 
% %                 n_label = n_label + 1;
% %     %             [x1, y1, x2, y2] = find_tip(label(1), tmp);
% %     %         
% %     %             record(5,ii) = x1;
% %     %             record(6,ii) = y1;
% %     %             record(7,ii) = x2;
% %     %             record(8,ii) = y2;
% %                 %display(record);
% %             end
% %         end
% %     end
% %     %display(record)
% %     valid_record_ind = record(1,:) > 0; % get valid record where label is not 0
% %     record = record(:,valid_record_ind);
% %     [Min, Imin] = min(record(3,:));  
% %     
% % %     valid_record_ind = record(3,:) < 0.9 | record(3,:) == Min;
% % %     record_valid = record(:, valid_record_ind);
% % %     
% % %     components_to_be_removed_ind = record(3,:) >= 0.9 & record(3,:)~= Min;
% % %     components_to_be_removed = record(:,components_to_be_removed_ind);
% % 
% %  %romove overlapping area that over 90% of the components   
% %           %overlapping area laerger than 90% remove
% % 
% % %     for ii = 1 : numel(components_to_be_removed(1,:))
% % %         layer_i = components_to_be_removed(4,ii);
% % %         label_i = components_to_be_removed(1,ii);
% % %         if layer_i~=record(4,Imin) && label_i ~= record(1, Imin)
% % %             tmp_layer_r = L(:,:,layer_i); %which layer to operate
% % %             pixel_to_remove = find(tmp_layer_r == label_i);
% % %             tmp_layer_r(pixel_to_remove) = 0;
% % %             L(:,:,layer_i) = tmp_layer_r;
% % %             
% % %         end
% % %     end
% %     
% % %    to_be_converted = find(record(3,:) >= 0.3);  
% %     k = 3 ;
% %     
% %    
% %     
% %     convert_to = record(1,Imin);
% %     to_be_converted = find_to_be_converted(Imin,record,L, k);
% %     %display( to_be_converted )
% %     for ii = 1 : numel(to_be_converted)  
% %         %if record(1,to_be_converted(ii)) ~= convert_to
% %            layer_location = record(4, to_be_converted(ii));
% %            label_num = record(1, to_be_converted(ii));
% %           % k_to_be_convert = find(L(:,:,layer_location) == label_num); % find all the label, to be convert.
% %            tmp_3 = L(:,:,layer_location );
% %            k_to_be_convert = find(tmp_3 == label_num);
% %            
% %            tmp_2 = L(:,:, record(4,Imin)); %record(4,Imin) the future layer of the label
% %            tmp_2(k_to_be_convert) = convert_to;
% %            
% %            tmp_3(k_to_be_convert) = 0;
% % 
% % 
% %            L(:,:, record(4,Imin)) = tmp_2; 
% %            L(:,:,layer_location) = tmp_3;
% %            if record(4,Imin) == layer_location
% %                display(100000000000000000000000000000000000000000000000000000000000)
% %            end
% %         %end
% %     end
% % 
% %      
% %     
% % end
% % 
% % 
% % Lmat = strcat(outputfolder,num2str(num),'_L.mat');
% % save(Lmat, 'L');
% % 
% % 
% % for i = 1 : 6
% %     tmp = L(:,:,i);
% %     for ii = 1 : max(max(tmp))
% %         canvas = zeros(size(tmp));
% %         ind = find(tmp == ii);
% %         canvas(ind) = 1;
% %         canvas = imdilate(canvas, ones(3,3));
% %         ind_v2 = find(canvas == 1);
% %         tmp(ind_v2) = ii;
% %     end
% %     L(:,:,i) = tmp;
% %     imwrite(label2rgb(L(:, :,i),'hsv', 'k', 'shuffle'), strcat(outputfolder,num2str(num),'_After_merge_check_',num2str(i),'.png')) ;
% % end
% % end
% % 
