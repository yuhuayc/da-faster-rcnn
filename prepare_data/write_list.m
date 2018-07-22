function write_list(cell_to_write, list_path)

fid=fopen(list_path,'w');
for i_w = 1:numel(cell_to_write)
    fprintf(fid,'%s\n',cell_to_write{i_w});
end

fclose(fid);
end