%a = imread('C:\work\Lab\CVPR2019\Mt_man\0_man.png');
clc
clear all;
nn = 7;
mode = 'man'; 
mode = 'sf';

a = imread(strcat('C:\work\Lab\CVPR2019\Mt_man\',num2str(nn),'_man.png'));
struct_mat = strcat(num2str(nn),'_L_gr.mat');
% struct_mat = strcat('0_L_sf.mat');
%a% = im2double(a);
[m,n,k] = size(a);
label = 1;
for_check = zeros(m,n);
for i = 1 : m
    for j = 1 : n
        %display (j)
        if a(i,j,1) ~= 0 || a(i,j,2) ~= 0 || a(i,j,3) ~= 0 
            canvas = zeros(m,n);
            %imshow(a)
            
            L(label).label = label;
            r = a(:,:,1);
            g = a(:,:,2);
            b = a(:,:,3);
            ind1 = find(r == a(i,j,1));
            ind2 = find(g == a(i,j,2));
            ind3 = find(b == a(i,j,3));
            index = intersect(ind1,ind2);
            index = intersect(index,ind3);
            [row,col] = ind2sub([m,n],index);
            

            
            canvas(index) = 1;
            di = imdilate(canvas,ones(5,5));
            
            skel_di = bwmorph(di,'thin',Inf);
            skel_or = bwmorph(canvas,'thin',Inf);
            
            %imshow(canvas)
            %imshow(di)
            %imshow(skel_di)
            %imshow(skel_or)
            
            
            
            di_ind = find(di);
            skel_di_ind = find(skel_di);
            skel_or_ind = find(skel_or);
            
            for_check(index) = 1;
            
            L(label).index = index;
            L(label).sk = skel_or_ind;
            L(label).di_ind = di_ind;
            L(label).sk_di = skel_di_ind;
            
            
            canvas_set_zero = uint8(1 - canvas);
            a(:,:,1) = a(:,:,1) .* canvas_set_zero;
            a(:,:,2) = a(:,:,2) .* canvas_set_zero;
            a(:,:,3) = a(:,:,3) .* canvas_set_zero;
            
            %%imshow(canvas);

            imshow(a)
            label = label + 1;
            display(label)
        end
    end
end
    imshow(for_check)
    save(struct_mat, 'L');
            
           