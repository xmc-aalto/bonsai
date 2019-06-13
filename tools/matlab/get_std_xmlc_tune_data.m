function [cv_split, cv_trn_X_Xf, cv_trn_X_Y, cv_tst_X_Xf, cv_tst_X_Y] = get_std_xmlc_tune_data( trn_X_Xf, trn_X_Y )
	
	cv_split = split_data(trn_X_Y,[],[],1);
	cv_trn_X_Xf = trn_X_Xf(:, cv_split==0);
	cv_trn_X_Y = trn_X_Y(:, cv_split==0);
	cv_tst_X_Xf = trn_X_Xf(:, cv_split==1);
	cv_tst_X_Y = trn_X_Y(:, cv_split==1);
end
