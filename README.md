# Rethinking Semantic-level Building Change Detection: Ensemble Learning and Dynamic Interaction

Here, we provide the pytorch implementation of the paper: Rethinking Semantic-level Building Change Detection: Ensemble Learning and Dynamic Interaction.

* For binary-level BCD: 
[binary-level](https://github.com/fengyuchao97/DIELNet/tree/main/CD_for_binary)
* For semantic-level BCD:
[semantic-level](https://github.com/fengyuchao97/DIELNet/tree/main/CD_for_semantic)

Details can be seen in each branch.

![data](./images/images/1.data.PNG)
Sample analysis of the available binary-level BCD datasets and the proposed method of constructing BCD samples through the BS dataset.

![task](./images/images/2.task.PNG)
Task descriptions for BCD at the binary-level and at the semantic-level.

![paradigm](./images/images/3.paradigm.PNG)
The independent training paradigm (left) and the proposed ensemble learning scheme (right) with multi-label and multi-task for BCD.

![model](./images/images/4.model.PNG)
The overall architecture of dynamic interaction ensemble learning network (DIELNet).

![FSPG](./images/images/5.FSPG.PNG)
Schematic diagram of our proposed frequency-swapped progressive generator (FSPG).

![sample](./images/images/6.Sample.PNG)
Examples presentation of available binary-level BCD samples and the proposed semantic-level BCD samples.

![distribute](./images/images/7.Distribute.PNG)
Sample distribution statistics for various datasets, where subplots (a), (d), and (g) present image-level distributions, subplots (b), (e), and (h) present percentage distributions of changed regions in each image, and subplots (c), (f), and (i) give the distributions of changed regions.

![distribute2](./images/images/9.Distribute2.PNG)
Sample distribution statistics for various datasets, where the left polar plot gives the distribution of binary-level changes, and the right subplots present the distribution of new buildings and demolitions, respectively.

![data_use](./images/images/8.Data.PNG)
The benchmark datasets used for experiments.

![size](./images/images/10.size.PNG)
Comprehensive comparisons of efficiency and performance. The left figure is efficiency comparisons of binary-level BCD on Inria-CD. The right figure shows the numerical analysis of demolished regions on WHU.

![binary](./images/images/11.binary_result.PNG)
Quantitative comparision of binary-level BCD on different datasets.

![binary2](./images/images/13.binary_result2.PNG)
Binary-level visual results of different methods equipped with ensemble learning on representative samples of LEVIR+, WHU and GZ.

![semantic](./images/images/12.semantic_result.PNG)
Quantitative comparision of semantic-level BCD on different datasets.

![semantic2](./images/images/14.semantic_result2.PNG)
Semantic-level visual results of different methods with or without ensemble learning on representative samples of WHU.

![semantic3](./images/images/15.semantic_result3.PNG)
Semantic-level visual results of different methods with ensemble learning on representative samples of LEVIR+ and GZ.

![spped](./images/images/16.speed.PNG)
Comparison of GPU memory usage (bar) and inference time (line) under batch size 64 and spatial size 256 × 256.

## Cite
If you use our method in your work please cite our paper:
* BibTex：


    @ARTICLE{10034787,
      author={Feng, Yuchao and Jiang, Jiawei and Xu, Honghui and Zheng, Jianwei},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Change Detection on Remote Sensing Images using Dual-branch Multi-level Inter-temporal Network}, 
      year={2023},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TGRS.2023.3241257}
    }


    @ARTICLE{9759285,
      author={Feng, Yuchao and Xu, Honghui and Jiang, Jiawei and Liu, Hao and Zheng, Jianwei},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={ICIF-Net: Intra-Scale Cross-Interaction and Inter-Scale Feature Fusion Network for Bitemporal Remote Sensing Images Change Detection}, 
      year={2022},
      volume={60},
      number={},
      pages={1-13},
      doi={10.1109/TGRS.2022.3168331}
    }
    

* Plane Text：
	
    Y. Feng, J. Jiang, H. Xu and J. Zheng, "Change Detection on Remote Sensing Images using Dual-branch Multi-level Inter-temporal Network," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2023.3241257.
    
    Y. Feng, H. Xu, J. Jiang, H. Liu and J. Zheng, "ICIF-Net: Intra-Scale Cross-Interaction and Inter-Scale Feature Fusion Network for Bitemporal Remote Sensing Images Change Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022, Art no. 4410213, doi: 10.1109/TGRS.2022.3168331.
    
    
    

