function run

% 	dset = 'EUR-Lex';
%    dset = 'Wiki10';
% 	dset = 'Ads-9M';
     dset = 'EUR-Lex-dense';
%    dset = 'Amazon-dense';

    %{
	param = [];
    param.num_tree = 3;
    param.max_leaf = 100;
    param.cent_th = 0.01;
    param.num_thread = 1;
	param.svm_iter = 50;
    param.beam_size = 10;
    param.discount = 1.0;
    param.septype = 0;
    %}

	param = [];
    param.num_tree = 3;
%     param.num_tree = 1;
    param.max_leaf = 10;
    param.cent_th = 0.01;
    param.num_thread = 1;
	param.svm_iter = 30;
    param.beam_size = 10;
    param.discount = 1.0;
    param.septype = 1;
    param.svm_th = 0.1;
    param.log_loss_coeff = 10.0;

	main( dset, param );
end
