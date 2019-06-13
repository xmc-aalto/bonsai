function print_struct(S,fid)
    fields = fieldnames(S);
    for i=1:numel(fields)
        fprintf(fid,'%s: %f\n',fields{i},getfield(S,fields{i}));
    end
end