function  to_be_converted = find_to_be_converted(Imin,record,L, k)  %return the index in the record of to be converted conponets
        to_be_converted = [];        

        kernel = ones(k);
        base_L = L(:,:, record(4, Imin));
        checkMap = zeros(size(base_L));
        
        base_label = record(1, Imin);
        base_pixel = find(base_L == base_label);
        checkMap(base_pixel) = 1;
        dir_map = checkMap;
        checkMap = imdilate(checkMap, kernel);
        [base_lt,base_rt] = find_tip(base_label, base_L);
        
        [~, num] = size(record);
        for i = 1 : num
            if Imin ~= i
               tmp_L = L(:,:, record(4, i));
               tmp_label = record(1, i);      
               
               checkMap_tmp = zeros(size(base_L));
               
               tmp_pixel = find(tmp_L == tmp_label);
               checkMap_tmp(tmp_pixel) = 1;
               dir_map_tmp = checkMap_tmp;
               checkMap_tmp = imdilate(checkMap_tmp, kernel);
               
               [tmp_lt,tmp_rt] = find_tip(tmp_label, tmp_L);
               
%                if(tmp_lt(2) == 16 || tmp_lt(2) == 17 || base_lt(2) == 16 ...
%                   ||base_lt(2) == 17)
%                                    
%                end
               % a is tmp
               % a's left in b && b'right in a: case 1
               % a's right in b && b' left in a: case 2
               
               % a's right in b && b's right in a case 3
               % a's left in b && b's left in a case 4
               
               % a's left in b && a's right in b
               % b's right in a && b's left in a 
               if((checkMap(tmp_lt(1),tmp_lt(2)) == 1 && checkMap_tmp(base_rt(1),base_rt(2)) == 1))
                   if(is_same_dir(dir_map, dir_map_tmp, base_rt, tmp_lt, 1))
                       to_be_converted = [to_be_converted, i];
                   end
                   
                   
               elseif(checkMap(tmp_rt(1), tmp_rt(2)) == 1 && checkMap_tmp(base_lt(1), base_lt(2)) == 1)
                   if(is_same_dir(dir_map, dir_map_tmp, base_lt, tmp_rt, 1))
                       to_be_converted = [to_be_converted, i];                       
                   end
               elseif(checkMap(tmp_rt(1), tmp_rt(2)) == 1 && checkMap_tmp(base_rt(1),base_rt(2)) == 1)
                   if(is_same_dir(dir_map, dir_map_tmp, base_rt, tmp_rt, 1))
                       to_be_converted = [to_be_converted, i];                       
                   end
               elseif(checkMap(tmp_lt(1), tmp_lt(2)) == 1 && checkMap_tmp(base_lt(1),base_lt(2)) == 1)
                   if(is_same_dir(dir_map, dir_map_tmp, base_rt, tmp_rt, 1))
                       to_be_converted = [to_be_converted, i];                       
                   end
               elseif(checkMap(tmp_lt(1), tmp_lt(2)) == 1 && checkMap(tmp_rt(1),tmp_rt(2)) == 1 )
                   to_be_converted = [to_be_converted, i];
               elseif(checkMap_tmp(base_lt(1), base_lt(2)) == 1 && checkMap_tmp(base_rt(1), base_rt(2)) == 1)
                   
                   to_be_converted = [to_be_converted, i];   
               end
               %subplot(2,2,1), imshow(checkMap)
               %subplot(2,2,2), imshow(checkMap_tmp)

                
            end
            
        end
        
        
        



end