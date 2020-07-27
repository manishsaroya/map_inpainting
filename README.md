# Tunnel-Network-Exploration-with-CNN-based-World-Predictions
**Note: I am in the process of cleaning up the code, should be done by the end of sept 2020. Contact me (saroyam@oregonstate.edu) if there is any problem with the code.**

Robotic exploration requires adaptively selecting navigation goals that result in the rapid discovery and mapping of an unknown world. In many real-world environments, subtle structural cues can provide insight about the unexplored world, which may be exploited by a decision maker to improve the
speed of exploration. In sparse subterranean tunnel networks,
these cues come in the form of topological features, such as
loops or dead-ends, that are often common across similar envi-
ronments. We propose a method for learning these topological
features using techniques borrowed from topological image
segmentation and image inpainting to learn from a database of
worlds. These world predictions then inform a frontier-based
exploration policy. Our simulated experiments with a set of
real-world mine environments and a database of procedurally-
generated artificial tunnel networks demonstrate a substantial
increase in the rate of area explored compared to techniques
that do not attempt to predict and exploit topological features
of the unexplored world.

## Requirements
- Python 3.6+
- Pytorch 0.4.1+

```
pip install -r requirements.txt
```

## Usage
Use synthetic data code to generate simulation data

### Preprocess 

```
python generate_data.py
```

### Train
```
python train.py
```
<!-- 
//### Fine-tune
//```
//CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --finetune --resume <checkpoint_name>
//``` -->
### Test
```
python test.py
```

## Results
Example trajectories for the Edgar Experimental Mine when using our CNN-based prefiction method and the closest first method. Robot (blue) navigates from the start (red) at bottom centre. At each timestep, the robot decides which frontier (orange) to navigate to next. 
![Results](results.png)

## References
- [1]: [pytorch inpainting with partial conv](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv)
