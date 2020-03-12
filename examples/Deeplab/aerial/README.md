
# Experiment on [https://project.inria.fr/aerialimagelabeling/](https://project.inria.fr/aerialimagelabeling/)


uploaded model on val result: 87.25%(official:68.7)
uploaded model on val result: 88.7%(official:--)

Val mIoU | Test mIoU
------------ | -------------
87.25 | 68.2
88.7(deeplabv2.naked.aerial.4gpu/model-35385) | ?
deeplabv2res101.aerial(88.07)|73.41
deeplabv2res101.aerial.shk(88.34)|?
deeplabv2res101.aerial.4gpu（88.67)|74.95
deeplabv2res101.aerial.shk.crop513.4gpu,(0.9,1,1.1 multi scale)(88.98)|?
deeplabv2res101.aerial.shk.crop513.4gpu,(0.7,0.9,1,1.1,1.3 multi scale)(89.14)|?
deeplabv2res101.aerial.shk.crop513.4gpu,(0.5,0.75,1,1.25,1.5 multi scale)(89.18)|75.33	
### models

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,no MSF,1GPU,bs10) | 87.17%,acc:95.58



### How Multi scale Fusion influence the result?

Arch | Val mIoU
------------ | -------------
(0.8,0.9,1,1.1,1.2) | 88.27,acc:96.02
(0.9,1,1.1) | 88.1,acc: 95.95
1           | 87.1,acc:95.58



### TODO

ResNext101+SENet+Mixup basic model will help greatly.
