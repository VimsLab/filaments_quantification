function [left_tip, right_tip] = find_tip(label, img)
    thined = bwmorph(img == label,'thin');
    [row_y, col_x] = find(bwmorph(thined,'endpoints'));
    num_of_endpoint = size(row_y);

    if  num_of_endpoint(1) > 2
       % display('num_of_end_point>2');
       % display('error');
        
    end
    
    if num_of_endpoint(1) <2
        %display('num_of_end_point<1');
        
       % pause;
        [row_y, col_x] = find(img==label);
        [x1,i] = min(col_x);
        y1 = row_y(i);
        left_tip = [y1, x1];
        [x2,j] = max(col_x);
        y2 = row_y(j);
        right_tip = [y2, x2];
    else
         [x1,i] = min(col_x);

         [x2,j] = max(col_x);
         if x1 == x2 
             y1 = max(row_y);
             y2 = min(row_y);
         else
             y1 = row_y(i);
             y2 = row_y(j);
         end
         
         left_tip = [y1, x1];
         
         right_tip = [y2, x2];
        
    end

    if(x1 > x2)
        %display('error');
       % pause;
        
    end
end