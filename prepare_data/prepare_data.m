%% An example of preparing data for da-faster-rcnn, the exmample is based on adaptation from Cityscapes to Foggy Cityscapes, other datasets can be prepared similarly.
%  
% yuhua chen <yuhua.chen@vision.ee.ethz.ch> 
% created on 2018.07.17

%% specify path
source_data_dir = 'CITYSCAPES_DIR';
target_data_dir = 'FOGGY_CITYSCAPES_DIR';

%% initialization
img_dir = 'VOCdevkit2007/VOC2007/JPEGImages';
sets_dir = 'VOCdevkit2007/VOC2007/ImageSets/Main';
annotation_dir = 'VOCdevkit2007/VOC2007/Annotations';

addpath VOCdevkit2007/VOCcode
mkdir(img_dir); mkdir(sets_dir); mkdir(annotation_dir);

%% organize images & prepare split list.

% process source train images
[~,cmd_output] = system(sprintf('find %s -name "*.png"', ...
    fullfile(source_data_dir,'leftImg8bit','train')));
file_names = strsplit(cmd_output); file_names = file_names(1:end-1);

source_train_list = cell(numel(file_names),1);
for i = 1:numel(file_names)
    im_name = strsplit(file_names{i},'/'); 
    im_name = strrep(im_name{end},'.png','');
    im_name = ['source_' im_name];
    
    img = imread(file_names{i});
    imwrite(img,fullfile(img_dir, [im_name '.jpg']));
    
    source_train_list{i} = im_name;
end

% process target train images.
[~,cmd_output] = system(sprintf('find %s -name "*_beta_0.02.png"', ...
    fullfile(target_data_dir,'leftImg8bit_foggy','train')));
file_names = strsplit(cmd_output); file_names = file_names(1:end-1);

target_train_list = cell(numel(file_names),1);
for i = 1:numel(file_names)
    im_name = strsplit(file_names{i},'/'); 
    im_name = strrep(im_name{end},'.png','');
    im_name = ['target_' im_name];
    
    img = imread(file_names{i});
    imwrite(img,fullfile(img_dir, [im_name '.jpg']));
    
    target_train_list{i} = im_name;
end

% process target test images
[~,cmd_output] = system(sprintf('find %s -name "*_beta_0.02.png"', ...
    fullfile(target_data_dir,'leftImg8bit_foggy','val')));
file_names = strsplit(cmd_output); file_names = file_names(1:end-1);

target_test_list = cell(numel(file_names),1);
for i = 1:numel(file_names)
    im_name = strsplit(file_names{i},'/'); 
    im_name = strrep(im_name{end},'.png','');
    im_name = ['target_' im_name];
    
    img = imread(file_names{i});
    imwrite(img,fullfile(img_dir, [im_name '.jpg']));
    
    target_test_list{i} = im_name;
end

% write the list
train_list = [source_train_list;target_train_list];
test_list = target_test_list;

write_list(train_list,fullfile(sets_dir,'trainval.txt'));
write_list(test_list,fullfile(sets_dir,'test.txt'));

%% prepare the annotation needed for training/testing.
load cityscapes_semantics
[~,lb_filter] = ismember(cityscapes_semantics,instance_semantics);

[~,cmd_output] = system(sprintf('find %s -name "*instanceIds.png"', ...
    fullfile(source_data_dir,'gtFine')));
file_names = strsplit(cmd_output); file_names = file_names(1:end-1);

for i = 1:numel(file_names)

    im_name = strsplit(file_names{i},'/'); 
    im_name = strrep(im_name{end},'_gtFine_instanceIds.png','_leftImg8bit');
    im_name = ['source_' im_name];

    im_inst = imread(file_names{i});
    im_lb = imread(strrep(file_names{i},'_gtFine_instanceIds.png','_gtFine_labelIds.png'));
    all_inst_id = setdiff(unique(im_inst),0);
    
    clear gt
    gt.boxes = zeros(numel(all_inst_id),4);
    gt.category = zeros(numel(all_inst_id),1);
    for i_inst = 1:numel(all_inst_id)
        inst_mask = (im_inst==all_inst_id(i_inst));
        gt.boxes(i_inst,:) = mask2box(inst_mask);
        inst_lb = unique(im_lb(inst_mask));
        gt.category(i_inst) = lb_filter(inst_lb);
    end
    gt.boxes = gt.boxes(gt.category~=0,:);
    gt.category = gt.category(gt.category~=0);
    
    clear save_var
    save_var.annotation.folder = 'VOC2007';
    save_var.annotation.filename = [im_name,'.jpg'];
    save_var.annotation.segmented = '0';
    save_var.annotation.size.width = num2str(size(im_inst,2));
    save_var.annotation.size.height = num2str(size(im_inst,1));
    save_var.annotation.size.depth = '3';
    
    for i_obj = 1:numel(gt.category)
        bbox = gt.boxes(i_obj,:);
        save_var.annotation.object(i_obj).bndbox.xmin = num2str(bbox(1));
        save_var.annotation.object(i_obj).bndbox.ymin = num2str(bbox(2));
        save_var.annotation.object(i_obj).bndbox.xmax = num2str(bbox(3));
        save_var.annotation.object(i_obj).bndbox.ymax = num2str(bbox(4));
        save_var.annotation.object(i_obj).name = ...
            (instance_semantics(gt.category(i_obj)));
        save_var.annotation.object(i_obj).difficult = '0';
        save_var.annotation.object(i_obj).truncated = '0';   
    end
    
    VOCwritexml(save_var,fullfile(annotation_dir,[im_name,'.xml']));
end

% for the target domain: Foggy Cityscapes, the
% annotation is the same, simply copy from Cityscapes

all_target_images = [target_train_list;target_test_list];
for i = 1:numel(all_target_images)
    im_name = all_target_images{i};
    source_name = strrep(strrep(im_name,'_foggy_beta_0.02',''),'target','source');
    copyfile(fullfile(annotation_dir,[source_name,'.xml']),...
        fullfile(annotation_dir,[im_name,'.xml']));
end
