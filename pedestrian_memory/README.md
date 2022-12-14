###preperation
####Dataset
RAPv1 : [project](http://www.rapdataset.com/)
####Path
Change the image path : dataset/pedes_attr/pedes.py

###Train and evaluation
Follow the installation protocol from [Microsoft recommenders](https://github.com/microsoft/recommenders)

Run following python files in order.

    (1) train_backbone.py
    (2) train_recommender.py
    (3) train_fusion.py

You can change the backbone type from #configspedes_baseline_rapv1.yaml

### Acknowledgements

KETI 시각상식 과제

### Reference Codes

- [Rethinking of PAR](https://github.com/valencebond/Rethinking_of_PAR)
- [Recommender system](https://github.com/microsoft/recommenders)

```
@article{jia2021rethinking,
  title={Rethinking of Pedestrian Attribute Recognition: A Reliable Evaluation under Zero-Shot Pedestrian Identity Setting},
  author={Jia, Jian and Huang, Houjing and Chen, Xiaotang and Huang, Kaiqi},
  journal={arXiv preprint arXiv:2107.03576},
  year={2021}
}
```