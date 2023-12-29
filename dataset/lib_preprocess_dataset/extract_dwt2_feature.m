function [xar] = extract_dwt2_feature(Input)
%EXTRACT_DWT2_FEATURE 此处显示有关此函数的摘要

 % Low-frequent image: xar
[xar,~,~,~]=dwt2(Input,'haar');

end

