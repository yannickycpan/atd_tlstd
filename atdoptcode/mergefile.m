function [] = mergefile(num_params)
%generate three files for each algorithm
algnames = {'TD','TO-TD','TO-ETD', 'ATD2nd'};

nalgs = length(algnames);
dirname = 'RMDPresults/jobs/';
combase = '_';
comprefix = 'mcarjob_';
%puddlejob_147_ATD2nd_PramNames.txt

paranames_suffix = '_PramNames.txt';
var_suffix = '_Var.txt';
mse_suffix = '_LC.txt';
% if there is algo name in the file name, use map to map from index to algo
% name

missedjobsid = [];

for i=0:nalgs-1    
    
%msenamefiles = dir(strcat(dirname,'*_', algnames{i+1},paranames_suffix));
%num_params = length(msenamefiles);
% disp('num of params is');
% num_params
newpre = 'RMDPresults/mcar_';

msename_file = strcat(newpre, algnames{i+1}, paranames_suffix);
fmerge_msename = fopen(msename_file, 'wt');

msevar_file = strcat(newpre, algnames{i+1}, var_suffix);
fmerge_msevar = fopen(msevar_file, 'wt');

mse_file = strcat(newpre, algnames{i+1}, mse_suffix);
fmerge_mse = fopen(mse_file, 'wt');

for curf = 0:num_params-1
    filename = strcat(dirname, comprefix, num2str(curf), combase, algnames{i+1},paranames_suffix);
    
    if exist(filename, 'file')
    fwrite(fmerge_msename, fileread(filename));
    
    filename = strcat(dirname, comprefix, num2str(curf), combase, algnames{i+1},var_suffix);
    fwrite(fmerge_msevar, fileread(filename));
    
    filename = strcat(dirname, comprefix, num2str(curf), combase, algnames{i+1},mse_suffix);
    fwrite(fmerge_mse, fileread(filename));
    elseif i == 0
       missedjobsid = [missedjobsid, curf];
    end
end
fclose(fmerge_msename);
fclose(fmerge_msevar);
fclose(fmerge_mse);
end

if length(missedjobsid)>0
fid = fopen('missid.txt','a');
for i=1:length(missedjobsid)
fprintf(fid, '%d,',missedjobsid(i));
end
fclose(fid);
end%end if missedjobsid

end
