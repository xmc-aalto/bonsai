function metric_data = get_tune_metric_data( tst_X_Y, inv_prop, metric ) 

	switch(metric)
		case {'precision_k','nDCG_k'}
			metric_data = {tst_X_Y, 5};

		case {'precision_wt_k','nDCG_wt_k'}
			metric_data = {tst_X_Y, inv_prop, 5};

		otherwise
			error(['invalid "metric" in ' mname]);
	end
end

