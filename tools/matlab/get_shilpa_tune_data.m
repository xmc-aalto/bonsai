function [cv_split, cv_trn_u_X_Xf, cv_trn_i_X_Xf, cv_trn_X_Y, cv_tst_u_X_Xf, cv_tst_i_X_Xf, cv_inc_tst_X_Y, cv_exc_tst_X_Y, Y_Yf] = get_shilpa_tune_data( trn_u_X_Xf, trn_i_X_Xf, trn_X_Y, Y_Yf, frac )
	
	cv_split = split_data(trn_X_Y,[],[],1);
	
	cv_trn_u_X_Xf = trn_u_X_Xf(:, cv_split==0);
	cv_trn_i_X_Xf = trn_i_X_Xf(:, cv_split==0);
	cv_trn_X_Y = trn_X_Y(:, cv_split==0);
	
	cv_tst_u_X_Xf = trn_u_X_Xf(:, cv_split==1);
	cv_tst_X_Y = trn_X_Y(:, cv_split==1);

	cv_inc_tst_X_Y = get_shilpa_inc_mat( cv_tst_X_Y, frac );
	cv_exc_tst_X_Y = cv_tst_X_Y - cv_inc_tst_X_Y;

	cv_tst_i_X_Xf = Y_Yf*cv_inc_tst_X_Y;
	prm.type = 'NORM_1';
	[~, cv_tst_i_X_Xf] = norm_features.fit_transform( cv_tst_i_X_Xf, prm );
end
