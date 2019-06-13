function bestparam = grid_search(fhandle, fdata, mhandle, mdata, param, cvparam, res_dir)
	fprintf('grid search starts\n\n');

    flds = fieldnames(cvparam);
    tcell = struct2cell(cvparam);
    parr = combvec(tcell{:});
    pn = size(parr,2);
    pval = zeros(1,pn);

 
    for i=1:pn
        copyparam = param;
        for j=1:numel(flds)
          copyparam = setfield(copyparam, flds{j}, parr(j,i));
        end
        
        fhandle( fdata{:}, copyparam, res_dir );
		load([res_dir filesep 'score_mat.mat']);
        pval(i) = mean(mhandle(score_mat, mdata{:}));

		fprintf('param:\n');
		disp(copyparam);
		fprintf('metric:\n');
		disp(pval(i));
		fprintf('\n\n');

		clear mex;
    end
 
    [~, imax] = max(pval, [], 2);
    bestparam = param;
    for j=1:numel(flds)
        setfield(bestparam, flds{j}, parr(j,imax));
    end

	fprintf('best param:\n');
	disp(bestparam);
	fprintf('metric:\n');
	disp(pval(imax));
	fprintf('\n\n');


	fprintf('grid search ends\n\n');

end
