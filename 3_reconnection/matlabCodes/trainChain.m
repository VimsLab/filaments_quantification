im1 = imread('0_1_predicted.png');
im2 = imread('0_2_predicted.png');
im3 = imread('0_3_predicted.png');
im4 = imread('0_4_predicted.png');
im5 = imread('0_5_predicted.png');
im6 = imread('0_6_predicted.png');
im7 = imread('0_7_predicted.png');

im1 = (im2double(im1)> 0.5);
im2 = (im2double(im2)> 0.5);
im3 = (im2double(im3)> 0.5);
im4 = (im2double(im4)> 0.5);
im5 = (im2double(im5)> 0.5);
im6 = (im2double(im6)> 0.5);
im7 = (im2double(im7)> 0.5);


im1 = bwmorph(im1,'bridge');
im1 = bwareaopen(im1,5);


im2 = bwmorph(im2,'bridge');
im2 = bwareaopen(im2,5);

im3 = bwmorph(im3,'bridge');
im3 = bwareaopen(im3,5);

im4 = bwmorph(im4,'bridge');
im4 = bwareaopen(im4,5);

im5 = bwmorph(im5,'bridge');
im5 = bwareaopen(im5,5);

im6 = bwmorph(im6,'bridge');
im6 = bwareaopen(im6,5);

b1 =  bwmorph(im1, 'endpoints');
searchSize = 10;
[row_y, col_x] = find(b1);

[L1,nL1] = bwlabel(im1);
imwrite(label2rgb(L1,'hsv', 'k', 'shuffle'), 'test_1.png');


[L2,nL2] = bwlabel(im2);

[L3,nL3] = bwlabel(im3);

[L4,nL4] = bwlabel(im4);

[L5,nL5] = bwlabel(im5);

[L6,nL6] = bwlabel(im6);

for i = 1:numel(col_x)
   if(row_y(i) - searchSize > 0 && row_y(i) + searchSize < 1524 && col_x(i) - searchSize > 0 && col_x(i) + searchSize < 1400)
    
%   window = disImage(yb(i) : yb(i) + 2 * searchSize, xb(i) + 2 * searchSize : xb(i) + 2 * searchSize);
    windowL1 = L1(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL2 = L2(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL3 = L3(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL4 = L4(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL5 = L5(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL6 = L6(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize); 
    
    

    
    overLap2 = windowL1 .* windowL2; 
    overLap3 = windowL1 .* windowL3; 
    overLap4 = windowL1 .* windowL4;    
    overLap5 = windowL1 .* windowL5; 
    overLap6 = windowL1 .* windowL6; 
    
    k2 = find(overLap2) ;  
    numb2 = numel(k2);

    k3 = find(overLap3) ;
    numb3 = numel(k3);
    

    k4 = find(overLap4);
    numb4 = numel(k4);

    k5 = find(overLap5);
    numb5 = numel(k5);    
    

    k6 = find(overLap6);
    numb6 = numel(k6);
    
    [M, I] = max([numb2, numb3, numb4, numb5, numb6]);
    
if I == 1 && M ~= 0   
    
    [row_o2, col_o2] = find(overLap2 > 0);
    if numel(row_o2) > 0
       converted_label = windowL2(row_o2(1),col_o2(1));
       convert_to = windowL1(row_o2(1),col_o2(1));
       ind2 = find(L2 == converted_label);
       L1(ind2) = convert_to;
       L2(ind2) = 0;
       
    end
end


if I == 2 && M ~= 0   
    [row_o3, col_o3] = find(overLap3 > 0);
    if numel(row_o3) > 0
       converted_label3 = windowL3(row_o3(1),col_o3(1));
       convert_to3 = windowL1(row_o3(1),col_o3(1));
       ind3 = find(L3 == converted_label3);
       L1(ind3) = convert_to3;
       L3(ind3) = 0;
       
    end
end
    
if I == 3 && M ~= 0       
 
    [row_o4, col_o4] = find(overLap4 > 0);
    if numel(row_o4) > 10
       converted_label4 = windowL4(row_o4(1),col_o4(1));
       convert_to4 = windowL1(row_o4(1),col_o4(1));
       ind4 = find(L4 == converted_label4);
       L1(ind4) = convert_to4;
       L4(ind4) = 0;
       
    end
    
end


 if I == 4 && M ~= 0      

    [row_o5, col_o5] = find(overLap5 > 0);
    if numel(row_o5) > 10
       converted_label5 = windowL5(row_o5(1),col_o5(1));
       convert_to5 = windowL1(row_o5(1),col_o5(1));
       ind5 = find(L5 == converted_label5);
       L1(ind5) = convert_to5;
       L5(ind5) = 0;
       
    end
 end
    
if I == 5 && M ~= 0       

    [row_o6, col_o6] = find(overLap6 > 0);
    if numel(row_o6) > 10
       converted_label6 = windowL6(row_o6(1),col_o6(1));
       convert_to6 = windowL1(row_o6(1),col_o6(1));
       ind6 = find(L6 == converted_label6);
       L1(ind6) = convert_to6;
       L6(ind6) = 0;
       
    end
end   

    
   end
end

imwrite(label2rgb(L2,'hsv','k','shuffle'),'test_2_L2_before.png')

b2 =  bwmorph(im2, 'endpoints');
% searchSize = 10;
[row_y, col_x] = find(b2);


for i = 1:numel(col_x)
   if(row_y(i) - searchSize > 0 && row_y(i) + searchSize < 1524 && col_x(i) - searchSize > 0 && col_x(i) + searchSize < 1400)
    
%   window = disImage(yb(i) : yb(i) + 2 * searchSize, xb(i) + 2 * searchSize : xb(i) + 2 * searchSize);
    windowL1 = L1(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL2 = L2(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL3 = L3(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL4 = L4(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL5 = L5(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL6 = L6(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize); 
    
    

    
    overLap2 = windowL2 .* windowL1; 
    overLap3 = windowL2 .* windowL3; 
    overLap4 = windowL2 .* windowL4;    
    overLap5 = windowL2 .* windowL5; 
    overLap6 = windowL2 .* windowL6; 
    
    k2 = find(overLap2) ;  
    numb2 = numel(k2);

    k3 = find(overLap3) ;
    numb3 = numel(k3);
    

    k4 = find(overLap4);
    numb4 = numel(k4);

    k5 = find(overLap5);
    numb5 = numel(k5);    
    

    k6 = find(overLap6);
    numb6 = numel(k6);
    
    [M, I] = max([numb2, numb3, numb4, numb5, numb6]);
    
if I == 1 && M ~= 0   
    
    [row_o2, col_o2] = find(overLap2 > 0);
    if numel(row_o2) > 0
       converted_label = windowL1(row_o2(1),col_o2(1));
       convert_to = windowL2(row_o2(1),col_o2(1));
       ind2 = find(L1 == converted_label);
       L2(ind2) = convert_to;
       L1(ind2) = 0;
       
    end
end


if I == 2 && M ~= 0   
    [row_o3, col_o3] = find(overLap3 > 0);
    if numel(row_o3) > 0
       converted_label3 = windowL3(row_o3(1),col_o3(1));
       convert_to3 = windowL2(row_o3(1),col_o3(1));
       ind3 = find(L3 == converted_label3);
       L2(ind3) = convert_to3;
       L3(ind3) = 0;
       
    end
end
    
if I == 3 && M ~= 0       
 
    [row_o4, col_o4] = find(overLap4 > 0);
    if numel(row_o4) > 10
       converted_label4 = windowL4(row_o4(1),col_o4(1));
       convert_to4 = windowL2(row_o4(1),col_o4(1));
       ind4 = find(L4 == converted_label4);
       L2(ind4) = convert_to4;
       L4(ind4) = 0;
       
    end
    
end


 if I == 4 && M ~= 0      

    [row_o5, col_o5] = find(overLap5 > 0);
    if numel(row_o5) > 10
       converted_label5 = windowL5(row_o5(1),col_o5(1));
       convert_to5 = windowL2(row_o5(1),col_o5(1));
       ind5 = find(L5 == converted_label5);
       L2(ind5) = convert_to5;
       L5(ind5) = 0;
       
    end
 end
    
if I == 5 && M ~= 0       

    [row_o6, col_o6] = find(overLap6 > 0);
    if numel(row_o6) > 10
       converted_label6 = windowL6(row_o6(1),col_o6(1));
       convert_to6 = windowL2(row_o6(1),col_o6(1));
       ind6 = find(L6 == converted_label6);
       L2(ind6) = convert_to6;
       L6(ind6) = 0;
       
    end
end   

    
   end
end

b3 =  bwmorph(im3, 'endpoints');
% searchSize = 10;
[row_y, col_x] = find(b3);


for i = 1:numel(col_x)
   if(row_y(i) - searchSize > 0 && row_y(i) + searchSize < 1524 && col_x(i) - searchSize > 0 && col_x(i) + searchSize < 1400)
    
%   window = disImage(yb(i) : yb(i) + 2 * searchSize, xb(i) + 2 * searchSize : xb(i) + 2 * searchSize);
    windowL1 = L1(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL2 = L2(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL3 = L3(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL4 = L4(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL5 = L5(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL6 = L6(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize); 
    
    

    
    overLap2 = windowL3 .* windowL1; 
    overLap3 = windowL3 .* windowL2; 
    overLap4 = windowL3 .* windowL4;    
    overLap5 = windowL3 .* windowL5; 
    overLap6 = windowL3 .* windowL6; 
    
    k2 = find(overLap2) ;  
    numb2 = numel(k2);

    k3 = find(overLap3) ;
    numb3 = numel(k3);
    

    k4 = find(overLap4);
    numb4 = numel(k4);

    k5 = find(overLap5);
    numb5 = numel(k5);    
    

    k6 = find(overLap6);
    numb6 = numel(k6);
    
    [M, I] = max([numb2, numb3, numb4, numb5, numb6]);
    
if I == 1 && M ~= 0   
    
    [row_o2, col_o2] = find(overLap2 > 0);
    if numel(row_o2) > 0
       converted_label = windowL1(row_o2(1),col_o2(1));
       convert_to = windowL3(row_o2(1),col_o2(1));
       ind2 = find(L1 == converted_label);
       L3(ind2) = convert_to;
       L1(ind2) = 0;
       
    end
end


if I == 2 && M ~= 0   
    [row_o3, col_o3] = find(overLap3 > 0);
    if numel(row_o3) > 0
       converted_label2 = windowL2(row_o3(1),col_o3(1));
       convert_to2 = windowL3(row_o3(1),col_o3(1));
       ind3 = find(L2 == converted_label2);
       L3(ind3) = convert_to2;
       L2(ind3) = 0;
       
    end
end
    
if I == 3 && M ~= 0       
 
    [row_o4, col_o4] = find(overLap4 > 0);
    if numel(row_o4) > 10
       converted_label4 = windowL4(row_o4(1),col_o4(1));
       convert_to4 = windowL3(row_o4(1),col_o4(1));
       ind4 = find(L4 == converted_label4);
       L3(ind4) = convert_to4;
       L4(ind4) = 0;
       
    end
    
end


 if I == 4 && M ~= 0      

    [row_o5, col_o5] = find(overLap5 > 0);
    if numel(row_o5) > 10
       converted_label5 = windowL5(row_o5(1),col_o5(1));
       convert_to5 = windowL3(row_o5(1),col_o5(1));
       ind5 = find(L5 == converted_label5);
       L3(ind5) = convert_to5;
       L5(ind5) = 0;
       
    end
 end
    
if I == 5 && M ~= 0       

    [row_o6, col_o6] = find(overLap6 > 0);
    if numel(row_o6) > 10
       converted_label6 = windowL6(row_o6(1),col_o6(1));
       convert_to6 = windowL3(row_o6(1),col_o6(1));
       ind6 = find(L6 == converted_label6);
       L3(ind6) = convert_to6;
       L6(ind6) = 0;
       
    end
end   

    
   end
end





b4 =  bwmorph(im4, 'endpoints');
% searchSize = 10;
[row_y, col_x] = find(b4);


for i = 1:numel(col_x)
   if(row_y(i) - searchSize > 0 && row_y(i) + searchSize < 1524 && col_x(i) - searchSize > 0 && col_x(i) + searchSize < 1400)
    
%   window = disImage(yb(i) : yb(i) + 2 * searchSize, xb(i) + 2 * searchSize : xb(i) + 2 * searchSize);
    windowL1 = L1(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL2 = L2(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL3 = L3(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL4 = L4(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL5 = L5(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL6 = L6(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize); 
    
    

    
    overLap2 = windowL4.* windowL1; 
    overLap3 = windowL4 .* windowL2; 
    overLap4 = windowL4 .* windowL3;    
    overLap5 = windowL4 .* windowL5; 
    overLap6 = windowL4 .* windowL6; 
    
    k2 = find(overLap2) ;  
    numb2 = numel(k2);

    k3 = find(overLap3) ;
    numb3 = numel(k3);
    

    k4 = find(overLap4);
    numb4 = numel(k4);

    k5 = find(overLap5);
    numb5 = numel(k5);    
    

    k6 = find(overLap6);
    numb6 = numel(k6);
    
    [M, I] = max([numb2, numb3, numb4, numb5, numb6]);
    
if I == 1 && M ~= 0   
    
    [row_o2, col_o2] = find(overLap2 > 0);
    if numel(row_o2) > 0
       converted_label = windowL1(row_o2(1),col_o2(1));
       convert_to = windowL4(row_o2(1),col_o2(1));
       ind2 = find(L1 == converted_label);
       L4(ind2) = convert_to;
       L1(ind2) = 0;
       
    end
end


if I == 2 && M ~= 0   
    [row_o3, col_o3] = find(overLap3 > 0);
    if numel(row_o3) > 0
       converted_label2 = windowL2(row_o3(1),col_o3(1));
       convert_to2 = windowL4(row_o3(1),col_o3(1));
       ind3 = find(L2 == converted_label2);
       L4(ind3) = convert_to2;
       L2(ind3) = 0;
       
    end
end
    
if I == 3 && M ~= 0       
 
    [row_o4, col_o4] = find(overLap4 > 0);
    if numel(row_o4) > 10
       converted_label4 = windowL3(row_o4(1),col_o4(1));
       convert_to4 = windowL4(row_o4(1),col_o4(1));
       ind4 = find(L3 == converted_label4);
       L4(ind4) = convert_to4;
       L3(ind4) = 0;
       
    end
    
end


 if I == 4 && M ~= 0      

    [row_o5, col_o5] = find(overLap5 > 0);
    if numel(row_o5) > 10
       converted_label5 = windowL5(row_o5(1),col_o5(1));
       convert_to5 = windowL4(row_o5(1),col_o5(1));
       ind5 = find(L5 == converted_label5);
       L4(ind5) = convert_to5;
       L5(ind5) = 0;
       
    end
 end
    
if I == 5 && M ~= 0       

    [row_o6, col_o6] = find(overLap6 > 0);
    if numel(row_o6) > 10
       converted_label6 = windowL6(row_o6(1),col_o6(1));
       convert_to6 = windowL4(row_o6(1),col_o6(1));
       ind6 = find(L6 == converted_label6);
       L4(ind6) = convert_to6;
       L6(ind6) = 0;
       
    end
end   

    
   end
end




b5 =  bwmorph(im5, 'endpoints');
% searchSize = 10;
[row_y, col_x] = find(b5);


for i = 1:numel(col_x)
   if(row_y(i) - searchSize > 0 && row_y(i) + searchSize < 1524 && col_x(i) - searchSize > 0 && col_x(i) + searchSize < 1400)
    
%   window = disImage(yb(i) : yb(i) + 2 * searchSize, xb(i) + 2 * searchSize : xb(i) + 2 * searchSize);
    windowL1 = L1(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL2 = L2(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL3 = L3(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL4 = L4(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL5 = L5(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL6 = L6(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize); 
    
    

    
    overLap2 = windowL5.* windowL1; 
    overLap3 = windowL5 .* windowL2;  %change
    overLap4 = windowL5 .* windowL3;    
    overLap5 = windowL5 .* windowL4; 
    overLap6 = windowL5 .* windowL6; 
    
    k2 = find(overLap2) ;  
    numb2 = numel(k2);

    k3 = find(overLap3) ;
    numb3 = numel(k3);
    

    k4 = find(overLap4);
    numb4 = numel(k4);

    k5 = find(overLap5);
    numb5 = numel(k5);    
    

    k6 = find(overLap6);
    numb6 = numel(k6);
    
    [M, I] = max([numb2, numb3, numb4, numb5, numb6]);
    
if I == 1 && M ~= 0   
    
    [row_o2, col_o2] = find(overLap2 > 0);
    if numel(row_o2) > 0
       converted_label = windowL1(row_o2(1),col_o2(1));
       convert_to = windowL5(row_o2(1),col_o2(1));
       ind2 = find(L1 == converted_label);
       L5(ind2) = convert_to;
       L1(ind2) = 0;
       
    end
end


if I == 2 && M ~= 0   
    [row_o3, col_o3] = find(overLap3 > 0);
    if numel(row_o3) > 0
       converted_label2 = windowL2(row_o3(1),col_o3(1));
       convert_to2 = windowL5(row_o3(1),col_o3(1));
       ind3 = find(L2 == converted_label2);
       L5(ind3) = convert_to2;
       L2(ind3) = 0;
       
    end
end
    
if I == 3 && M ~= 0       
 
    [row_o4, col_o4] = find(overLap4 > 0);
    if numel(row_o4) > 10
       converted_label4 = windowL3(row_o4(1),col_o4(1));
       convert_to4 = windowL5(row_o4(1),col_o4(1));
       ind4 = find(L3 == converted_label4);
       L5(ind4) = convert_to4;
       L3(ind4) = 0;
       
    end
    
end


 if I == 4 && M ~= 0      

    [row_o5, col_o5] = find(overLap5 > 0);
    if numel(row_o5) > 10
       converted_label5 = windowL4(row_o5(1),col_o5(1));
       convert_to5 = windowL5(row_o5(1),col_o5(1));
       ind5 = find(L4 == converted_label5);
       L5(ind5) = convert_to5;
       L4(ind5) = 0;
       
    end
 end
    
if I == 5 && M ~= 0       

    [row_o6, col_o6] = find(overLap6 > 0);
    if numel(row_o6) > 10
       converted_label6 = windowL6(row_o6(1),col_o6(1));
       convert_to6 = windowL5(row_o6(1),col_o6(1));
       ind6 = find(L6 == converted_label6);
       L5(ind6) = convert_to6;
       L6(ind6) = 0;
       
    end
end   

    
   end
end


b6 =  bwmorph(im6, 'endpoints');
% searchSize = 10;
[row_y, col_x] = find(b6);


for i = 1:numel(col_x)
   if(row_y(i) - searchSize > 0 && row_y(i) + searchSize < 1524 && col_x(i) - searchSize > 0 && col_x(i) + searchSize < 1400)
    
%   window = disImage(yb(i) : yb(i) + 2 * searchSize, xb(i) + 2 * searchSize : xb(i) + 2 * searchSize);
    windowL1 = L1(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL2 = L2(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL3 = L3(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL4 = L4(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL5 = L5(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize);
    windowL6 = L6(row_y(i) - searchSize: row_y(i) + searchSize, col_x(i) - searchSize: col_x(i) + searchSize); 
    
    

    
    overLap2 = windowL6.* windowL1; 
    overLap3 = windowL6 .* windowL2; 
    overLap4 = windowL6 .* windowL3;    
    overLap5 = windowL6 .* windowL4; 
    overLap6 = windowL6 .* windowL5; 
    
    k2 = find(overLap2) ;  
    numb2 = numel(k2);

    k3 = find(overLap3) ;
    numb3 = numel(k3);
    

    k4 = find(overLap4);
    numb4 = numel(k4);

    k5 = find(overLap5);
    numb5 = numel(k5);    
    

    k6 = find(overLap6);
    numb6 = numel(k6);
    
    [M, I] = max([numb2, numb3, numb4, numb5, numb6]);
    
if I == 1 && M ~= 0   
    
    [row_o2, col_o2] = find(overLap2 > 0);
    if numel(row_o2) > 0
       converted_label = windowL1(row_o2(1),col_o2(1));
       convert_to = windowL6(row_o2(1),col_o2(1));
       ind2 = find(L1 == converted_label);
       L6(ind2) = convert_to;
       L1(ind2) = 0;
       
    end
end


if I == 2 && M ~= 0   
    [row_o3, col_o3] = find(overLap3 > 0);
    if numel(row_o3) > 0
       converted_label2 = windowL2(row_o3(1),col_o3(1));
       convert_to2 = windowL6(row_o3(1),col_o3(1));
       ind3 = find(L2 == converted_label2);
       L6(ind3) = convert_to2;
       L2(ind3) = 0;
       
    end
end
    
if I == 3 && M ~= 0       
 
    [row_o4, col_o4] = find(overLap4 > 0);
    if numel(row_o4) > 10
       converted_label4 = windowL3(row_o4(1),col_o4(1));
       convert_to4 = windowL6(row_o4(1),col_o4(1));
       ind4 = find(L3 == converted_label4);
       L6(ind4) = convert_to4;
       L3(ind4) = 0;
       
    end
    
end


 if I == 4 && M ~= 0      

    [row_o5, col_o5] = find(overLap5 > 0);
    if numel(row_o5) > 10
       converted_label5 = windowL4(row_o5(1),col_o5(1));
       convert_to5 = windowL6(row_o5(1),col_o5(1));
       ind5 = find(L4 == converted_label5);
       L6(ind5) = convert_to5;
       L4(ind5) = 0;
       
    end
 end
    
if I == 5 && M ~= 0       

    [row_o6, col_o6] = find(overLap6 > 0);
    if numel(row_o6) > 10
       converted_label6 = windowL5(row_o6(1),col_o6(1));
       convert_to6 = windowL6(row_o6(1),col_o6(1));
       ind6 = find(L5 == converted_label6);
       L6(ind6) = convert_to6;
       L5(ind6) = 0;
       
    end
end   

    
   end
end









imwrite(label2rgb(L1,'hsv', 'k', 'shuffle'), 'test_2_L1.png');
imwrite(label2rgb(L2,'hsv', 'k', 'shuffle'), 'test_2_L2.png');
imwrite(label2rgb(L3,'hsv', 'k', 'shuffle'), 'test_2_L3.png');
imwrite(label2rgb(L4,'hsv', 'k', 'shuffle'), 'test_2_L4.png') ;
imwrite(label2rgb(L5,'hsv', 'k', 'shuffle'), 'test_2_L5.png') ;
imwrite(label2rgb(L6,'hsv', 'k', 'shuffle'), 'test_2_L6.png') ;


imshow(label2rgb(L3,'hsv', 'k', 'shuffle')) ;
