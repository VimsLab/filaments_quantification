function component = group(label, layer, tmp_l,L)
    canvas = zeros(size(tmp_1));
    for ii = 1 : numel(label_l)
       [pixels, num] = find(tmp_l == label_l);
       canvas(pixels) = 1;
       canvas = bwareaopen(canvas,4);
       searching_area = imdialte(canvas, ones(3,3))
    end


end