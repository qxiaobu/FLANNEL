# FLANNEL

## Data Prepare
### Data Collect
1. Download CCX data: from https://github.com/ieee8023/covid-chestxray-dataset, put them into original_data/covid-chestxray-dataset-master
2. Download KCX data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia, put them into original_data/chest_xray
### Data Preprocess
1. extract data from CCX: data_preprocess/get_covid_data_dict.py 
2. extract data from KCX: data_preprocess/get_kaggle_data_dict.py
3. reorganize CCX&KCX data to generate 5-folder cross-validation expdata: data_preprocess/extract_exp_data_crossentropy.py

## Model Training
### Base-modeler Learning
FLANNEL/ensemble_step1.py for 5 base-modeler learning [InceptionV3, Vgg19_bn, ResNeXt101, Resnet152, Densenet161]

(E.g. python ensemble_step1.py --arch InceptionV3)

### ensemble-model Learning
FLANNEL/ensemble_step2_ensemble_learning.py



@misc{qiao2020flannel, \r\n
      title={FLANNEL: Focal Loss Based Neural Network Ensemble for COVID-19 Detection},  \r\n
      author={Zhi Qiao and Austin Bae and Lucas M. Glass and Cao Xiao and Jimeng Sun}, \r\n
      year={2020}, \r\n
      eprint={2010.16039}, \r\n
      archivePrefix={arXiv}, \r\n
      primaryClass={eess.IV} \r\n
}
