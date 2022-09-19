
DL_final - v3 Third Set (no rotations/noise)
==============================

This dataset was exported via roboflow.ai on December 25, 2020 at 12:11 AM GMT

It includes 4717 images.
Garbage are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 250x250 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random brigthness adjustment of between -35 and +35 percent


