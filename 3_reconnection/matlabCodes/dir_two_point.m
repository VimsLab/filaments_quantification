function  k = dir_two_point(ps,pe)%start point to end point
                                  %ps [y_col, x_row]
                                  %return a vector [y1-y2, x1-2]
    k = [(pe(1) - ps(1)),(pe(2) - ps(2))];

end