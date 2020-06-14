function orient = check_dir_between2pts(p1,p2) % this function only applies to image matrix; angle calculated is between [-180,180]

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