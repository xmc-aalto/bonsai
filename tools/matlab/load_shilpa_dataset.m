function [trn_u_X_Xf, trn_i_X_Xf, trn_X_Y, tst_u_X_Xf, tst_i_X_Xf, inc_tst_X_Y, exc_tst_X_Y, Y_Yf, inv_prop] = load_shilpa_dataset(dset, frac)
	global EXP_DIR RES_DIR;

	[X_Xf, X_Y, Y_Yf] = load_dataset(dset);
	split = load_split(dset);

	trn_u_X_Xf = X_Xf(:,split==0);
	trn_X_Y = X_Y(:,split==0);
	tst_u_X_Xf = X_Xf(:,split==1);
	tst_X_Y = X_Y(:,split==1);

	prm.type = 'NORM_1';

	trn_i_X_Xf = Y_Yf*trn_X_Y;
	[~,trn_i_X_Xf] = norm_features.fit_transform(trn_i_X_Xf,prm);

	inv_prop = inv_propensity_wrap(trn_X_Y, dset);
    
    inc_tst_X_Y = get_shilpa_inc_mat( tst_X_Y, frac );
    exc_tst_X_Y = tst_X_Y - inc_tst_X_Y;

	tst_i_X_Xf = Y_Yf*inc_tst_X_Y;
	[~,tst_i_X_Xf] = norm_features.fit_transform(tst_i_X_Xf,prm);
end
