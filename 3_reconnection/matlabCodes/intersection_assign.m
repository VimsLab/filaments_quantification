im1 = imread('0_1_predicted.png');
im2 = imread('0_2_predicted.png');
im3 = imread('0_3_predicted.png');
im4 = imread('0_4_predicted.png');
im5 = imread('0_5_predicted.png');
im6 = imread('0_6_predicted.png');
im7 = imread('0_7_predicted.png');

stack = cat(3, im1, im2, im3, im4, im5, im6, im7);

stack = (im2double(stack)> 0.5);
%% do some cleaning here.  Q: should I do the same thing to the 7th layer?
  %clearn some little gaps and get rid of little fragments
for i = 1 : 7 
     stack(:,:, i) = bwmorph(stack(:,:,i),'bridge'); 
     stack(:,:, i) = bwareaopen(stack(:,:,i),5);
end

%% Find all the intersection in im7. which can be a lot though.

b1 =  bwmorph(stack(:,:,7), 'branchpoints');

% se = strel('disk', 3);  % trying to reduce the number of unnecssary
% brachpoints here.but nevermind, it will reduce the pricision of branch
% points. Sacrifice some speed then. 
% b1 = imdilate(b1, se);

b2 = zeros(size(b1));
[Li, nLi] = bwlabel(b1);
b2 = b1;
% for i = 1 : nLi
%    [row_y, col_x] = find(Li == i);
%    row_center = round(mean(row_y));
%    col_center = round(mean(col_x));
%    b2(row_center, col_center) = i;
% end

%% Get rid of the intersction of im7, run disconected components on it.
b2 = b2 > 0;
mask = imdilate(b2, ones(3,3));
mask = mask > 0;
disImage = (stack(:,:,7) & ~mask);

[L, Ln] = bwlabel(disImage);

%% reconnection

searchSize = 15;

% zero pad
paded = padarray(stack, [searchSize, searchSize], 0, 'both');
b2_paded = padarray(b2, [searchSize, searchSize], 0, 'both');
L_paded = padarray(L, [searchSize, searchSize], 0, 'both');
% find b2

[row_y, col_x] = find(b2_paded);

%get the windows
for i = 1 : numel(row_y)
    disp(i);
    window = paded(row_y(i) - searchSize : row_y(i) + searchSize, col_x(i) - searchSize : col_x(i) + searchSize, : );
    windowL = L_paded(row_y(i) - searchSize:  row_y(i) + searchSize, col_x(i) - searchSize : col_x(i) + searchSize); 
    label_win_pos = find(windowL > 0); 
    label_win = windowL(label_win_pos);                    %get the label in the window
    layer_label_win = zeros(size(label_win));                 %find out which layer this label belongs to
    layer_label_win = cat(2, label_win, layer_label_win);    % structure |label|layer|
    
    for ii = 1 : numel(label_win)
        if layer_label_win(ii,1) > 0 
        layer_label_win(ii, 2) = whichLayer(label_win(ii), window, 6, windowL);  % whichLayer(label, windows, x, windowL)     
        
        end
    end
    
    for iii = 1 : 6
        ind = find(layer_label_win(:,2) == iii);
        assigned_label = min(layer_label_win(ind,1));
        for iiii = 1 : numel(ind)
            L_paded(find(L_paded == layer_label_win(ind(iiii)))) = assigned_label;          
            
        end
    end
    
end

imshow(label2rgb(L_paded,'hsv', 'k', 'shuffle')) ;

