function main(input, output)

%inputFolder = 'C:\work\Lab\Stromules\Reconnection\FromVims\intersection\manmadeTestingData\';
%outputFolder = '.\testing418\';
display('here')
%inputFolder = '../input/';
%outputFolder = '../output/';
inputFolder = input;
outputFolder = output;
display(inputFolder)
display(outputFolder)

files = dir (strcat(inputFolder,'*_branches_merge.png'));
display(files)

for i = 1: length(files)
    fileNames = files(i).name;
    name = split(fileNames, '_branches_merge.png');
   % tic
   %TipSearch_v2(i-1,'output_alexData_1/','alexData_1/')
    name = name{1};
    TipSearch_v3(name,outputFolder,inputFolder); %output folder, % inputFolder
   
   %  TipSearch_v2(i-1,'output_input_v2_modified_11_14/','input/')
    %TipSearch_v2(400,'C:\work\Lab\CVPR2019\PlanktothrixDemoSet\final\','C:\work\Lab\CVPR2019\PlanktothrixDemoSet\concat\')
    %TipSearch_v2(i-1,'output_input_di5_v2_modified_11_14/','test_on_mt_di5/')
%    TipSearch_v2(i-1,'output_test_on_mt_di5_v2_test/','test_on_mt_di5/')
  %  TipSearch_v2(i-1,'output_test_on_mt_di5_v3_test/','test_on_mt_di5/')
   % TipSearch_v2(i,'C:\work\Lab\CVPR2019\PlanktothrixDemoSet\final\','C:\work\Lab\CVPR2019\PlanktothrixDemoSet\pre_sep\')
 %   TipSearch_v2(i,'C:\work\Lab\CVPR2019\ROAD\Final_v2_bt_distance10\','C:\work\Lab\CVPR2019\ROAD\use_our_prediction\')
    
    %search size 50
end

end