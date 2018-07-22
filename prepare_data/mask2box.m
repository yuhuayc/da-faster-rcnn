function box = mask2box( mask )

%% start
assert(length(unique(mask))==2);
x_cods = find(sum(mask,1)>0);
y_cods = find(sum(mask,2)>0);

box = [min(x_cods),min(y_cods),max(x_cods),max(y_cods)];

end