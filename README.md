# PCAT

***

The implementation for the KDD 2025 paper "Paper-Level Computerized Adaptive Testing for High-Stakes Examination via Multi-Objective Optimization".

***

# 💻 Requirements	

```python
torch==2.3.1+cu118
numpy==1.26.4
pandas==2.2.2
pymoo==0.6.1.5
scikit-learn==1.5.1
```

Please install all the dependencies listed in the `requirements.txt` file by running the following command:

```bash
pip install -r requirements.txt
```

# 🛠️ File distribution

To make it easier for other scholars to use our code, we'll explain the purpose and location of each file in the root directory:  

1.**data** 
Because the dataset is too large, we compress it when we upload it. So before using this code, you should unzip the dataset file using the following command:
```shell
unzip data.zip
```

This folder is used to store the data set after densification of the original data set,as well as the densification handler.  
You can use the following command to process existing data sets or custom data sets.
```shell
python generate_densedata.py
```
After the densification operation, you will obtain dataset like "Dense_X" file.

2.**Data**
This is where the training and testing data object construction code is stored

3.**model**
This folder stores the code for cognitive diagnostic model NCD, related pre-trained models, etc.

4.**strategy**
Various selection algorithm related code is stored.

5.**scripts**
Contains various permission-related code.

# 🛠️ Experiments

Firstly, you need

> cd scripts/run_code

To run our code, run the following:
```shell
python run_code/main.py
```

If you want to change the dataset, just replace the parameter of `--dataset` with the name of your target dataset such as "Dense_Assistment17" or "Dense_MOOCRadar" and so on.


If you want to change the test length, just replace the parameter of `--test_length` with the number of your target length such as 20 or 10 and so on.

After running the program, the result file *"Dense_X_output.txt"* will be generated in the run_code directory.

>Noting: In order to provide a more convenient baseline test, our framework is also equipped with other benchmark methods such as random, kil, NCAT, etc. You can change this in the `strategy_list` object in main.py.

# Reference

Mingjia Li, Junkai Tong, Yiyang Huang, Yifei Ding, Hong Qian and Aimin Zhou "Paper-Level Computerized Adaptive Testing for High-Stakes Examination via Multi-Objective Optimization" In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025.

## Bibtex
```
@inproceedings{li2025pcat,
 author = {Mingjia Li, Junkai Tong, Yiyang Huang, Yifei Ding, Hong Qian, Aimin Zhou},
 booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
 title = {Paper-Level Computerized Adaptive Testing for High-Stakes Examination via Multi-Objective Optimization},
 year = {2025},
 address = {Toronto, Canada}
}
```
 
