function [lk,rk] = get_dir(canvas, lt, rt, distance) %15recomended
    canvas = logical(canvas);
    D_dir_lt = bwdistgeodesic(canvas,lt(2), lt(1)); % column and then row
    D_dir_rt = bwdistgeodesic(canvas,rt(2), rt(1)); % column and then row
    if max(D_dir_lt(:)) == 0 || isinf(max(D_dir_lt(:)))
        end_lt = lt;     
    else
        [end_y, end_x] = find(D_dir_lt == min(max(D_dir_lt(:)),distance));   
        end_lt = [end_y(1),end_x(1)];
    end
    lk = dir_two_point(lt, end_lt);
    
    if max(D_dir_rt(:)) == 0 || isinf(max(D_dir_lt(:)))
        end_rt = rt;     
    else
        [end_y, end_x] = find(D_dir_rt == min(max(D_dir_rt(:)),distance));  
        
        
        end_rt = [end_y(1),end_x(1)];
        
        

    end

    rk = dir_two_point(rt,end_rt);
end