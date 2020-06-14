function index = draw_line(rt1, lt2, size_mat) 
    mat = zeros(size_mat);
    x = [rt1(2) lt2(2)];
    y = [rt1(1) lt2(1)];
    
    nPoints = max(abs(diff(x)), abs(diff(y)))+1;% Number of points in line
    rIndex = round(linspace(y(1), y(2), nPoints));  % Row indices
    cIndex = round(linspace(x(1), x(2), nPoints));  % Column indices
    index = sub2ind(size(mat), rIndex, cIndex);     % Linear indices
end