function [update_state, L_struct] = reconnect(label, new_six, L_struct, search_size, update_state)
if L_struct(label).exist == 0
    update_state = 0;
    return;
end


display(label);
if label == 76

end
    threshold = 0.85;
    %conditions overlap
    related_label = [];
    canvas = zeros(size(new_six(:,:,1)));
    lt = L_struct(label).lt;
    rt = L_struct(label).rt;
    

    canvas(lt(1), lt(2)) = 1;
    canvas(rt(1), rt(2)) = 1;
    canvas = imdilate(canvas, ones(search_size));
    
    for i = 1 : 6
    
        unique_label = unique(canvas .* new_six(:,:,i));
        related_label = union(related_label, unique_label);
        
    end
    related_label = unique(related_label);
    related_label = related_label(related_label > 0);
    
    
    canvas2 = zeros(size(new_six(:,:,1))); 
    for t = 1 : numel(related_label)
        canvas2(L_struct(related_label(t)).index) = t;
    end
    canvas3 = canvas2 + canvas;
    %imshow(label2rgb(canvas3,'hsv', 'k', 'shuffle'),[]) ;
    
    
    
    base_label = label;
    base = L_struct(label);
    
    record = [];
    
    
    for i = 1 : numel(related_label)
        if L_struct(related_label(i)).exist == 1
            if related_label(i) ~= base_label
                
                canvas4 = zeros(size(new_six(:,:,1)));
                canvas4(L_struct(related_label(i)).index) = 1;
                canvas4 = imdilate(canvas4, ones(3));
                canvas4= (canvas2 + canvas + canvas4);
             %   imshow(label2rgb(canvas4,'hsv', 'k', 'shuffle'),[]) ;
                
                
                %% overlapping case;
                target = L_struct(related_label(i));
                num_overlap = numel(intersect(base.index,target.index));
                if num_overlap ~= 0
                    base_percent = num_overlap / numel(base.index);
                    target_percent = num_overlap / numel(target.index);
                    [min_percent,the_min] = min([target_percent,base_percent]);
                    [large_percent,the_large] = max([target_percent,base_percent]);
                    if large_percent > threshold
                        if the_large == 1 %1 is the target. the target
                           L_struct(related_label(i)).exist = 0;
                        elseif the_large == 2
                           L_struct(label) = target;
                           L_struct(related_label(i)).exist = 0;
                        end
                        
                        update_state = 1;
                        break;
                        
                    else  %%%% wrong here, shouldnt think about target . should think about the end of the mask.
                          base_ind_lt = sub2ind(size(canvas), ...
                                        base.lt(1),base.lt(2));
                          base_ind_rt = sub2ind(size(canvas), ...
                                        base.rt(1),base.rt(2));
                          tar_ind_lt = sub2ind(size(canvas), ...
                                        target.lt(1),target.lt(2));
                          tar_ind_rt = sub2ind(size(canvas), ...
                                        target.rt(1),target.rt(2));
                         %1 a's left in b && b'right in a: case 1
                         if( ismember(base_ind_lt, target.ind_dilate) && ismember(tar_ind_rt, base.ind_dilate))
                             k1 = base.ld;
                             k2 = target.rd;
    
                             dir_result = dot(k1,k2)/(norm(k1)*norm(k2));
                             if dir_result < -0.70
%                                  new_six(new_six ==related_label(i)) = 0;
%                                  new_six(new_six ==related_label(i)) = 0;
                                 base.index = union(base.index, target.index);
                                 base.ind_dilate = union(base.ind_dilate,target.ind_dilate);
                                 canvas_2 = zeros(size(canvas));
                                 canvas_2(base.index) = 1;
                                 [base.lt, base.rt] = find_tip(1,canvas_2); % find tip new tip
                                 
                                 %%%%%% CHANGE PARA
                                 [base.ld, base.rd] = get_dir(canvas_2, base.lt, base.rt, 10);
                                 target.exist = 0;
                                 L_struct(label) = base;
                                 L_struct(related_label(i)) = target;
                                 update_state = 1;
                                 break;
                         
                             end

                         %2 a's right in b && b' left in a: case 2
                         elseif(ismember(base_ind_rt, target.ind_dilate) && ismember(tar_ind_lt, base.ind_dilate))  
                             k1 = base.rd;
                             k2 = target.ld;
    
                             dir_result = dot(k1,k2)/(norm(k1)*norm(k2));
                             if dir_result < -0.70
%                                  new_six(new_six ==related_label(i)) = 0;
%                                  new_six(new_six ==related_label(i)) = 0;
                                 
                                 base.index = union(base.index, target.index);
                                 base.ind_dilate = union(base.ind_dilate,target.ind_dilate);
                                 canvas_2 = zeros(size(canvas));
                                 canvas_2(base.index) = 1;
                                 [base.lt, base.rt] = find_tip(1,canvas_2); % find tip new tip
                                 
                                 %%%%%% CHANGE PARA
                                 [base.ld, base.rd] = get_dir(canvas_2, base.lt, base.rt, 10);
                                 target.exist = 0;
                                 L_struct(label) = base;
                                 L_struct(related_label(i)) = target;
                                 update_state = 1;
                                 break;
                             end
                             
                          %3 a's right in b && b's right in a case 3
                         elseif(ismember(base_ind_rt, target.ind_dilate) && ismember(tar_ind_rt, base.ind_dilate))
                             k1 = base.rd;
                             k2 = target.rd;
    
                             dir_result = dot(k1,k2)/(norm(k1)*norm(k2));
                             if dir_result < -0.70
%                                  new_six(new_six ==related_label(i)) = 0;
%                                  new_six(new_six ==related_label(i)) = 0;
                                 base.index = union(base.index, target.index);
                                 base.ind_dilate = union(base.ind_dilate,target.ind_dilate);
                                 
                                 canvas_2 = zeros(size(canvas));
                                 canvas_2(base.index) = 1;
                                 [base.lt, base.rt] = find_tip(1,canvas_2); % find tip new tip
                                 
                                 %%%%%% CHANGE PARA
                                 [base.ld, base.rd] = get_dir(canvas_2, base.lt, base.rt, 10);
                                 target.exist = 0;
                                 L_struct(related_label(i)) = target;
                                 L_struct(label) = base;
                                 update_state = 1;
                                 break;
                             end
                         %4 a's left in b && b's left in a case 4
                         elseif(ismember(base_ind_lt, target.ind_dilate) && ismember(tar_ind_lt, base.ind_dilate))
                             k1 = base.ld;
                             k2 = target.ld;
    
                             dir_result = dot(k1,k2)/(norm(k1)*norm(k2));
                             if dir_result < -0.70
%                                  new_six(new_six ==related_label(i)) = 0;
%                                  new_six(new_six ==related_label(i)) = 0;
                                 base.index = union(base.index, target.index);
                                  base.ind_dilate = union(base.ind_dilate,target.ind_dilate);
                                 
                                 canvas_2 = zeros(size(canvas));
                                 canvas_2(base.index) = 1;
                                 [base.lt, base.rt] = find_tip(1,canvas_2); % find tip new tip
                                 
                                 %%%%%% CHANGE PARA
                                 [base.ld, base.rd] = get_dir(canvas_2, base.lt, base.rt, 10);
                                 target.exist = 0;
                                 L_struct(label) = base;
                                 L_struct(related_label(i)) = target;
                                 
                                 update_state = 1;
                                 break;
                             end
                          %4 a's left in b && a's right in b
                          %5 b's right in a && b's left in a 
                         end
%                          elseif((ismember(base_ind_lt, target.ind_dilate) && ismember(base_ind_rt, target.ind_dilate)) ...
%                                 || (ismember(tar_ind_lt, base.ind_dilate) && ismember(tar_ind_rt, base.ind_dilate))) % this case could be redundant as long as the overlaping part above is working.
%                             
% %                                  new_six(new_six ==related_label(i)) = 0;
% %                                  new_six(new_six ==related_label(i)) = 0;
%                                  base.index = union(base.index, target.index);
%                                   base.ind_dilate = union(base.ind_dilate,target.ind_dilate);
%                                  
%                                  canvas_2 = zeros(size(canvas));
%                                  canvas_2(base.index) = 1;
%                                  [base.lt, base.rt] = find_tip(1,canvas_2); % find tip new tip
%                                  
%                                  %%%%%% CHANGE PARA
%                                  [base.ld, base.rd] = get_dir(canvas_2, base.lt, base.rt, 10);
%                                  target.exist = 0;
%                                  L_struct(label) = base;
%                                  L_struct(related_label(i)) = target;
%                                  
%                                  update_state = 1;
%                                  break;                             
%                          end                    
                    end
                   %end of overlapping case;   
                else
                   %% numOfoverlapping is 0. Check tip.
                  base_ind_lt = sub2ind(size(canvas), ...
                                base.lt(1),base.lt(2));
                  base_ind_rt = sub2ind(size(canvas), ...
                                base.rt(1),base.rt(2));
                  tar_ind_lt = sub2ind(size(canvas), ...
                                target.lt(1),target.lt(2));
                  tar_ind_rt = sub2ind(size(canvas), ...
                                target.rt(1),target.rt(2));
                  canvas_distance = logical(canvas);
                  D_lt = bwdistgeodesic(canvas_distance, base_ind_lt);
                  D_rt = bwdistgeodesic(canvas_distance, base_ind_rt);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                  bt_angle = -0.5;
                  bt_distance = 10; %%%parameter seting
                  vD = zeros(4,4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                 
                  
                  

                  
                  LL = D_lt(tar_ind_lt); %1 % only consider tip dir, didn't think about tip to tip dir.  % soloved
                  

                  
                  k1 = base.ld;
                  k2 = target.ld; 
                  dir_LL = dot(k1,k2)/(norm(k1)*norm(k2));              
                  
                  k_LL = dir_two_point(base.lt, target.lt); %only tip to tip
                  dir_bt_LL = dot(k_LL,k1)/(norm(k_LL)*norm(k1)); %dir of end of base and dir of tip to tip  
                  
                  if dir_bt_LL > bt_angle
                        disLL_x = norm(k_LL) * dir_bt_LL;
                        disLL_y = sqrt(norm(k_LL)^2 - disLL_x^2);
                        if disLL_x < (bt_distance + 1) && disLL_y < (bt_distance + 1)
                            dir_bt_LL = -1;
                        end
                  end
 %%%%%%%%%%%%%%%%%%                 
                  
                  
                  LR = D_lt(tar_ind_rt); %2
                  k1 = base.ld;
                  k2 = target.rd; 
                  dir_LR = dot(k1,k2)/(norm(k1)*norm(k2));                     
                  k_LR = dir_two_point(base.lt, target.rt);  
                  dir_bt_LR = dot(k_LR,k1)/(norm(k_LR)*norm(k1)); %dir of end of base and dir of tip to tip   
                  
                  if dir_bt_LR > bt_angle
                        disLR_x = norm(k_LR) * dir_bt_LR;
                        disLR_y = sqrt(norm(k_LR)^2 - disLR_x^2);
                        if disLR_x < (bt_distance + 1) && disLR_y < (bt_distance + 1)
                            dir_bt_LR = -1;
                        end
                  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                  
                  
                  
                  RL = D_rt(tar_ind_lt); %3
                  
                  k1 = base.rd;
                  k2 = target.ld; 
                  dir_RL = dot(k1,k2)/(norm(k1)*norm(k2)); 
                  k_RL = dir_two_point(base.rt, target.lt);
                  dir_bt_RL = dot(k_RL,k1)/(norm(k_RL)*norm(k1)); %dir of end of base and dir of tip to tip 
                  
                  if dir_bt_RL > bt_angle
                        disRL_x = norm(k_RL) * dir_bt_RL;
                        disRL_y = sqrt(norm(k_RL)^2 - disRL_x^2);
                        if disRL_x < (bt_distance + 1) && disRL_y < (bt_distance + 1)
                            dir_bt_RL = -1;
                        end
                  end
                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                  
                  RR = D_rt(tar_ind_rt); %4
                  
                  
                  k1 = base.rd;
                  k2 = target.rd; 
                  dir_RR = dot(k1,k2)/(norm(k1)*norm(k2));    
                  k_RR = dir_two_point(base.rt, target.rt);
                  dir_bt_RR = dot(k_RR,k1)/(norm(k_RR)*norm(k1)); %dir of end of base and dir of tip to tip   
                  
                  if dir_bt_RR > bt_angle
                        disRR_x = norm(k_RR) * dir_bt_RR;
                        disRR_y = sqrt(norm(k_RR)^2 - disRR_x^2);
                        if disRR_x < (bt_distance + 1) && disRR_y < (bt_distance + 1)
                            dir_bt_RR = -1;
                        end
                  end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  
                  
                  vD_dis = [LL, LR, RL, RR];
                  vD_bt = [dir_bt_LL, dir_bt_LR, dir_bt_RL, dir_bt_RR];
                  vD_dir = [dir_LL,dir_LR, dir_RL, dir_RR];
             
                  vD_case = [1,2,3,4];
                  
                  vD(1,:) = vD_dis; %distance
                  vD(2,:) = vD_dir; %direction
                  vD(3,:) = vD_case; %which case. LL, LR, RL, RR
                  vD(4,:) = vD_bt;
                  
                  valid_ind = vD(2,:) < -0.70; %
                  vD = vD(:, valid_ind);
                  
                  
                  valid_ind = vD(4,:) < - 0.5; %less than 60 degrees.
                  
                  
                  vD = vD(:, valid_ind);
                  if (isempty(vD))
                      continue;
                  end
                  
%following is the distance consideration. 
%we dont need it probablly;

%                   min_value = min(vD(1,:));
%                   [dis_close] = find(vD(1,:) == min_value); %closest 
%                   vD = vD(:, dis_close);            
%                   
%                   if isnan(min_value) || isinf(min_value)
%                       continue;
%                   else
%                       min_dir = min(vD(2,:));
%                       dir_close = find(vD(2,:) == min_dir);
%                       vD = vD(:, dir_close);
%                       record(i).vD = vD;
%                       record(i).label = related_label(i);                        
%                   end

%following is the angle
                  
                  [nonNan] = find(~isnan(vD(1,:))); 
                  vD = vD(:, nonNan);   
                  [nonInf] = find(~isinf(vD(1,:))); 
                  vD = vD(:, nonInf);    
                  
%                   if isnan(min_value) || isinf(min_value)
%                       continue;
%                   else
%                       min_dir = min(vD(2,:));
%                       dir_close = find(vD(2,:) == min_dir);
%                       vD = vD(:, dir_close);
                      record(i).vD = vD;
                      record(i).label = related_label(i);                        
%                   end
                  
                  
                  
                   
                   
                   
                end
            
            end
        end
                
    end
    
    
    if update_state ~= 1
        if ~isempty(record)
            num_of_record = numel(record);
            which_label = Inf;
            distance = Inf;
            angle = Inf;
            ttt = Inf; %tip to tip angle
            which = Inf;
%             for j = 1 : num_of_record
%                 tmp_vD = record(j).vD; %%
%                 if isempty(tmp_vD)
%                     continue;
%                 end
%                 % only assume that two label has 4 tips and no two tips
%                 % with the same distance
%                 for jj = 1 : numel(tmp_vD(1, :))
%                     if tmp_vD(1,jj) < distance
%                         distance = tmp_vD(1,jj);
%                         which_label = record(j).label;
%                         which = tmp_vD(3, jj);
%                         angle = tmp_vD(2, jj);
%                         
%                     elseif tmp_vD(1,jj) == distance
%                         if tmp_vD(2,jj) < angle
%                             distance = tmp_vD(1,jj);
%                             which_label = record(j).label;
%                             which = tmp_vD(3, jj);
%                             angle = tmp_vD(2, jj);
%                         end
%                    
%                     end
%                 end
%             end
            for j = 1 : num_of_record
                tmp_vD = record(j).vD; % record(j) vD max four possible tips
                if isempty(tmp_vD)
                    continue;
                end
                % only assume that two label has 4 tips and no two tips
                % with the same distance
                for jj = 1 : numel(tmp_vD(1, :))
                    if tmp_vD(2,jj) < angle
                        distance = tmp_vD(1,jj);
                        which_label = record(j).label;
                        which = tmp_vD(3, jj);
                        angle = tmp_vD(2, jj);
                        ttt = tmp_vD(4, jj);
                    elseif tmp_vD(2,jj) == angle
                        if tmp_vD(4,jj) < ttt
                            distance = tmp_vD(1,jj);
                            which_label = record(j).label;
                            which = tmp_vD(3, jj);
                            angle = tmp_vD(2, jj);
                            ttt = tmp_vD(4, jj);
                        elseif tmp_vD(4,jj) == ttt
                            if tmp_vD(1,jj) < distance
                             	 distance = tmp_vD(1,jj);
                                 which_label = record(j).label;
                                which = tmp_vD(3, jj);
                                angle = tmp_vD(2, jj);
                                ttt = tmp_vD(4, jj);
                            end
                        end
                   
                    end
                end
            end

            
            if which_label ~= Inf
                tar = L_struct(which_label);
                if which == 1
                    %LL
                    gap_ind = draw_line(base.lt, tar.lt,size(canvas));
                elseif which == 2
                    %LR
                    gap_ind = draw_line(base.lt, tar.rt,size(canvas));                  
                elseif which == 3
                    %RL
                    gap_ind = draw_line(base.rt, tar.lt,size(canvas));
                elseif which == 4
                    %RR
                    gap_ind = draw_line(base.rt, tar.rt,size(canvas));
                end
                         
                base.index = union(base.index, tar.index);
                base.index = union(base.index, gap_ind);
                
                %%%!!!!!!!!!!!!!!
                   %missing gap_ind_dilate
                %%%!!!!!!!!!!!!

                
                 canvas_2 = zeros(size(canvas));
                canvas_2(base.index) = 1;
                 [base.lt, base.rt] = find_tip(1,canvas_2); % find tip new tip
                  base.ind_dilate = (imdilate(canvas_2, ones(3,3))>0);
                 %%%%%% CHANGE PARA
                 [base.ld, base.rd] = get_dir(canvas_2, base.lt, base.rt, 10);
                 tar.exist = 0;
                 L_struct(label) = base;
                 L_struct(which_label) = tar;
                 update_state = 1;
            end
        end
    end
    
    
    
    %
end