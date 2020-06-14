function layer = whichLayer(label, windows, x, windowL)
     index = find(windowL == label);
     max_num = 0;
     layer = 1;
     for i = 1 : x
         tempLayer = windows(:,:,i);
         corresponding_ele = tempLayer(index);
         num_of_ele = sum(corresponding_ele > 0);
         if num_of_ele > max_num
            max_num = num_of_ele;
            layer = i;
         end
     end  
end