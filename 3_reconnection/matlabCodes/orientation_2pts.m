% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function orient = orientation_2pts(p1,p2) % this function only applies to image matrix; angle calculated is between [-180,180]

if     p2(2)>p1(2) && p2(1)==p1(1)
    orient = 0;
elseif p2(2)<p1(2) && p2(1)==p1(1)
    orient = -180;
elseif p2(2)==p1(2) && p2(1)>p1(1)
    orient = -90;
elseif p2(2)==p1(2) && p2(1)<p1(1)
    orient = 90;
elseif p2(2)<p1(2)
    orient = 180*atan(   (p1(1)-p2(1))  /   (p2(2)-p1(2))   )/pi + sign(  p1(1)-p2(1)  )*180;
else
    orient = 180*atan(   (p1(1)-p2(1))  /   (p2(2)-p1(2))   )/pi;
end

end 

% this code has been tested using 8 different points


