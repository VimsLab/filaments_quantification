% function L = reLabel(old_six)
%     L = load('L12.mat');
%     L = L.L;
    old_six = L;
    
    %% Find overlaping area
% overLap = zeros(size(L(:,:,1)));
% %dilate_for_overlap = imdilate(stack, ones(5,5));
% for i = 1 : 6
%     for ii = i + 1 : 6
%     overLap = ((L(:,:,i) & L(:,:,ii)) | overLap);
%     %overLap = ((dilate_for_overlap(:,:,i) & dilate_for_overlap(:,:,ii)) | overLap);
%     end
%     overLap = bwareaopen(overLap, 5);
% end
%%
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
            imshow(label2rgb(label_canvas,'hsv', 'k', 'shuffle'),[])
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

                update_L(canvas > 0) = label_num;
                label_num = label_num + 1;
                imshow(label2rgb(update_L,'hsv', 'k', 'shuffle'),[]);
            end
            
        end
        new_six(:,:,i) = update_L;
        
    end
    save('new_six12.mat', 'new_six');
    save('L12_struct.mat', 'L_struct');
    
% end



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