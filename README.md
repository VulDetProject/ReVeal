# Deep Learning based Vulnerability Detection:Are We There Yet? 

### Code repository for the study

In this study, we empirically study different existing Deep Learning Based Vulnerability Detection techniques for real world vulnerabilities. 
We test the feasibility of existing techniques in two different datasets. 
1. [Part of Devign dataset](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF) (often referred to as FFMpeg+Qemu dataset in the project). 
2. [Our Collected vulnerabilities from Chrome and Debian issue trackers](https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy) (Often referred as Chrome+Debian or Verum dataset in this project).


To download data 

```
cd data;
bash get_data.sh
```

To download (some of) pretrained models
```
cd models;
bash get_models.sh
```

### Processing new data
Some of the tools in this study can be used for a new datasets. In order for doing that, we use [Joern]() for parsing the C code in this repository.   
```bash
cd code-slicer/joern;
bash build.sh
```
Once the build is successful, go to the folder you want to perform your experiment, create a folder named `raw_code` and create every functions in separate C files. 
We followed the custom to file names `<name>_<VUL>.c`, wehre the `<VUL>` is the Vulnerability identifier of the  function (0 for benign, 1 for vulnerable).

1. You have to extract the slices from the parsed code. Modify the [data_processing/extract_slices.ipynb](data_processing/extract_slices.ipynb) for extracting slice. 
This will generate a file `<data_name>_full_data_with_slices.json` in your data directory. 

2. Run [data_processing/create_ggnn_data.py](data_processing/create_ggnn_data.py) for formatting data into different formats.

3. Update [data_processing/full_data_prep_script.ipynb](data_processing/full_data_prep_script.ipynb) to input to the GGNN.

### Running GGNN. 

1. Clone our implemetation of Devign from [here](https://github.com/saikat107/Devign.git).
2. Use the following parameters `"node_features"` as `"--node_tag"`, `"graph"` as `--graph_tag`, and `targets` as `--label_tag`.
3. User `--save_after_ggnn` flag for saving the data after processing through GGNN.

### To try ReVeal pipeline as a whole, 
The running APIs are exposed by [this file](Vuld_SySe/representation_learning/api_test.py). Moddify the parameters to fit your need.

To try ReVeal on Chrome+Debian(Verum) dataset,
```bash
cd Vuld_SySe/representation_learning;
bash run_verum.sh
```

To try ReVeal on Devign dataset,
```bash
cd Vuld_SySe/representation_learning;
bash run_devign.sh
```

We include different scripts for running other models (_i.e._ VulDeePecker, SySeVR, Draper) under  `scripts/` and `real_data_scripts/` folders.

## Acknoledgements.

We are using several different components from the state-of-the-art research. Please cite accordingly to pay due attributes and credits to the authors.
1. If you use Code-Slicer portion from this repository, please cite the following
```latex
@inproceedings{yamaguchi2014modeling,
  title={Modeling and discovering vulnerabilities with code property graphs},
  author={Yamaguchi, Fabian and Golde, Nico and Arp, Daniel and Rieck, Konrad},
  booktitle={2014 IEEE Symposium on Security and Privacy},
  pages={590--604},
  year={2014},
  organization={IEEE}
}
```

2. If you use Devign, please cite,
```latex
@inproceedings{zhou2019devign,
  title={Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks},
  author={Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10197--10207},
  year={2019}
}
```

3. If you refer to empirical finding reported in the paper, please cite our pre-print as
```latex
@article{chakraborty2020deep,
  title={Deep Learning based Vulnerability Detection: Are We There Yet?},
  author={Chakraborty, Saikat and Krishna, Rahul and Ding, Yangruibo and Ray, Baishakhi},
  journal={arXiv preprint arXiv:2009.07235},
  year={2020}
}
```
