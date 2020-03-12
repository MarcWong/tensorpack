# Hourglass

# MPII Result 

Method|cherry picked PCKh
------------ | -------------
hourglass.scale5.stage1.adam2.5e-4| 59.8
hourglass.scale5.stage4.adam2.5e-4|61.1
hourglass.scale5.stage8.adam2.5e-4| 62.98
hourglass.scale5.stage4.adam1e-3| 56.13
hourglass.scale10.stage4.adam2.5e-4.always|79.2
hourglass.scale5.stage4.adam2.5e-4.always.newlr|80
hourglass.scale20.stage4.adam2.5e-4.always.newlr|stopped,too long,4 hours x 20 epoches|
hourglass.scale10.stage8.adam2.5e-4.always.newlr|stopped,too long, 4hours x 20 epoches||
hourglass.scale5.stage8.adam2.5e-4.always.newlr|74|
hourglass.scale10.stage4.adam2.5e-4.always.newlr|79.6|


# Question

* why PCKh is unstable during training?
* why shoulder pckh larger than 1?
* why cannot validation after training?
