function bestparam = sequential_search(fhandle, fdata, mhandle, mdata, param, cvparam, res_dir)
	fprintf('sequential search starts\n\n');
	
    flds = fieldnames(cvparam);

	bestparam = param;

	for i=1:numel(flds)
		fld = flds{i};
		fprintf('validating %s\n\n',fld);
		pvec = getfield(cvparam,fld);
		pval = zeros(1,numel(pvec));

		for j=1:numel(pvec)
			copyparam = bestparam;
			copyparam = setfield(copyparam, fld, pvec(j));
			fhandle(fdata{:}, copyparam, res_dir);
			load([res_dir filesep 'score_mat.mat']);
			pval(j) = mean(mhandle(score_mat, mdata{:}));

			fprintf('param:\n');
			disp(copyparam);
			fprintf('metric:\n');
			disp(pval(j));
			fprintf('\n\n');

			clear mex;
		end

		[~, jmax] = max(pval, [], 2);
		bestparam = setfield(bestparam, flds{i}, pvec(jmax));
	end

	fprintf('best param:\n');
	disp(bestparam);
	fprintf('\n\n');


	fprintf('sequential search ends\n\n');
end
