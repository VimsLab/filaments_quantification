function flag = is_same_dir(dir_map, dir_map_tmp, base_tip, tmp_tip, which_case)
    if which_case == 1 %base_tip on the left. tmp_tip on the right
    over_lap = dir_map & dir_map_tmp;
    over_lap = imdilate(over_lap,ones(19));
    
    dir_map = dir_map & over_lap;
    dir_map_tmp = dir_map_tmp & over_lap;
    
%     [base_epts_y,base_epts_x] = find(bwmorph(dir_map,'endpoints'));
%     [tmp_epts_y, tmp_epts_x] = find(bwmorph(dir_map_tmp,'endpoints'));
    %%%
    %[base_end_x,ibase] = min((base_epts_x ~= base_tip(2)));
    %[base_end_x,ibase] = min(base_epts_x);
    %base_end_y = base_epts_y(ibase);
    %%%%
    
    D_dir = bwdistgeodesic(dir_map,base_tip(2), base_tip(1)); % column and then row
    if max(D_dir(:)) == 0
        base_end_x = base_tip(2);
        base_end_y = base_tip(1);
        
    else
        [base_end_y, base_end_x] = find(D_dir == max(D_dir(:)));   
    end
    
    
   % I can defiitely speed up the program here. 
    
    k1 = [(base_end_y - base_tip(1)),(base_end_x - base_tip(2))];
    
%     %[tmp_end_x, itmp] = max(tmp_epts_x(tmp_epts_x ~= tmp_tip(2)));
%     [tmp_end_x, itmp] = max(tmp_epts_x);
%     tmp_end_y = tmp_epts_y(itmp);

    D_dir_tmp = bwdistgeodesic(dir_map_tmp,tmp_tip(2), tmp_tip(1)); % column and then row
    if max(D_dir_tmp(:)) == 0
        tmp_end_x = base_tip(2);
        tmp_end_y = base_tip(1);
        
    else
        [tmp_end_y, tmp_end_x] = find(D_dir_tmp == max(D_dir_tmp(:)));   
    end
    
    k2 = [(tmp_end_y - tmp_tip(1)), (tmp_end_x - tmp_tip(2))];
    
    k1 = mean(k1,1);
    k2 = mean(k2,1);
    
    C = dot(k1,k2)/(norm(k1)*norm(k2));
    
        if C < -0.75
            flag = true; 
        else
            flag = false;
        end
    end

% %     
% %     elseif which_case == 2 %base_tip on the right. tmp_tip on the left
% %    
% %     over_lap = dir_map & dir_map_tmp;
% %     over_lap = imdilate(over_lap,ones(19));
% %     
% %     dir_map = dir_map & over_lap;
% %     dir_map_tmp = dir_map_tmp & over_lap;
% %     
% %     [base_epts_y,base_epts_x] = find(bwmorph(dir_map,'endpoints'));
% %     [tmp_epts_y, tmp_epts_x] = find(bwmorph(dir_map_tmp,'endpoints'));
% %     
% %     [base_end_x,ibase] = max(base_epts_x);
% %     base_end_y = base_epts_y(ibase);
% %     
% %     k1 = [(base_end_y - base_tip(1)),(base_end_x - base_tip(2))];
% %     
% %     [tmp_end_x, itmp] = min(tmp_epts_x);
% %     tmp_end_y = tmp_epts_y(itmp);
% %     
% %     k2 = [(tmp_end_y - tmp_tip(1)), (tmp_end_x - tmp_tip(2))];
% %     
% %     C = dot(k1,k2);
% %     
% %         if C < 0
% %             flag = true; 
% %         else
% %             flag = false;
% %         end
% %         
% %     elseif which_case == 3 %base_tip on the right. tmp_tip on the left
% %    
% %     over_lap = dir_map & dir_map_tmp;
% %     over_lap = imdilate(over_lap,ones(19));
% %     
% %     dir_map = dir_map & over_lap;
% %     dir_map_tmp = dir_map_tmp & over_lap;
% %     
% %     [base_epts_y,base_epts_x] = find(bwmorph(dir_map,'endpoints'));
% %     [tmp_epts_y, tmp_epts_x] = find(bwmorph(dir_map_tmp,'endpoints'));
% %     
% %     [base_end_x,ibase] = min(base_epts_x);
% %     base_end_y = base_epts_y(ibase);
% %     
% %     k1 = [(base_end_y - base_tip(1)),(base_end_x - base_tip(2))];
% %     
% %     [tmp_end_x, itmp] = min(tmp_epts_x);
% %     tmp_end_y = tmp_epts_y(itmp);
% %     
% %     k2 = [(tmp_end_y - tmp_tip(1)), (tmp_end_x - tmp_tip(2))];
% %     
% %     C = dot(k1,k2);
% %     
% %         if C < 0
% %             flag = true; 
% %         else
% %             flag = false;
% %         end   
% %         
% %     elseif which_case == 4%base_tip on the right. tmp_tip on the left
% %    
% %     over_lap = dir_map & dir_map_tmp;
% %     over_lap = imdilate(over_lap,ones(19));
% %     
% %     dir_map = dir_map & over_lap;
% %     dir_map_tmp = dir_map_tmp & over_lap;
% %     
% %     [base_epts_y,base_epts_x] = find(bwmorph(dir_map,'endpoints'));
% %     [tmp_epts_y, tmp_epts_x] = find(bwmorph(dir_map_tmp,'endpoints'));
% %     
% %     [base_end_x,ibase] = max(base_epts_x);
% %     base_end_y = base_epts_y(ibase);
% %     
% %     k1 = [(base_end_y - base_tip(1)),(base_end_x - base_tip(2))];
% %     
% %     [tmp_end_x, itmp] = max(tmp_epts_x);
% %     tmp_end_y = tmp_epts_y(itmp);
% %     
% %     k2 = [(tmp_end_y - tmp_tip(1)), (tmp_end_x - tmp_tip(2))];
% %     
% %     C = dot(k1,k2)/ (norm(k1)*norm(k2));
% %     
% %         if C < - 0.75    %45 degree, opposite direction.
% %             flag = true; 
% %         else
% %             flag = false;
% %         end 
% %         
% %         
% %         
% %     end
    
    
    
    
    
    

end