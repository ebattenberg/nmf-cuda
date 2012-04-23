%% write test matrices to binary files
% values are store in column-major order

% nmf: X = W*H

% target X matrix
X = [1 2 3; 4 5 6; 7 8 9];
% initial W matrix 
W = [1 2; 3 4; 5 6; 7 8];
% initial H matrix
H = [1 2 3; 4 5 6]; 



fid = fopen([path 'X.bin'],'w');


fwrite(fid,size(X),'int');
count = fwrite(fid,X(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);



fid = fopen([path 'W.bin'],'w');

fwrite(fid,size(W),'int');
count = fwrite(fid,W(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);

fid = fopen([path 'H.bin'],'w');

fwrite(fid,size(H),'int');
count = fwrite(fid,H(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);



