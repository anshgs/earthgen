# EarthGen: Generating the World from Top-Down Views

[[Paper]](https://arxiv.org/abs/XXXX.XXXXX) 

[[Interactive Demo]](https://earthgen.github.io)

In this work, we present a novel method for extensive multi-scale generative terrain modeling. At the core of our model is a cascade of superresolution diffusion models that can be combined to produce consistent images across multiple resolutions. Pairing this concept with a tiled generation method yields a scalable system that can generate thousands of square kilometers of realistic Earth surfaces at high reso- lution. We evaluate our method on a dataset collected from Bing Maps and show that it outperforms super-resolution baselines on the extreme super-resolution task of 1024× zoom. We also demonstrate its ability to create diverse and coherent scenes via an interactive gigapixel-scale generated map. Finally, we demonstrate how our system can be extended to enable novel content creation applications including controllable world generation and 3D scene generation


## Running EarthGen
### Dataset
``scraper.py`` contains a script to collect satellite image pyramids from the Bing Maps API at random coordinates across the map, and buckets them based on the maximum zoom available. ``scraper_maps.py`` can be run afterward to augment the collected pyramids with road map data as well for training a controlnet. The specific coordinates used for the datasets in our paper are detailed in the coordinates folder in lat/lon format. 

### Training
``train_superres.py`` contains the code to fine tune a StableDiffusion x4 Upscaler given the dataset created above. ``train_controlnet.py`` extends the previous to fine tune a controlnet on top of the x4 Upscaler. An unconditional model can be trained by passing in blank images to the lr_image components of ``train_superres`` instead of the actual low resolution conditioning. 

### Inference
``multi_layer_no_tile.ipynb`` shows a minimal workflow to load and run a 1024x super-resolution on a single trajectory. ``generate_large_image.py`` contains a script to generate dense large-scale super-resolved images in a similar manner to our interactive demo. ``demo_controlnet.ipynb`` shows a minimal workflow to run the 1024x super-resolution on a single trajectory with an additional conditioning input. 

### Evaluation
``validation2048.py`` can be used to generate validation image stacks based on provided low resolution inputs. ``fid_kid_metrics.ipynb`` demonstrates metric evaluations on those stacks and computes FID and KID.

### Model Download
Fine-tuned checkpoints of the models used to generate the results for our paper have been uploaded to HuggingFace [here](https://huggingface.co/Earthgen/models). 10gen contains the base layer module as well as a map-conditioned controlnet add-on, while the rest of the folders in the format {lr}to{hr} include the corresponding super-resolution modules from zoom lr to zoom hr. 

## Citation
```
@misc{sharma2024earthgengeneratingworldtopdown,
      title={EarthGen: Generating the World from Top-Down Views}, 
      author={Ansh Sharma and Albert Xiao and Praneet Rathi and Rohit Kundu and Albert Zhai and Yuan Shen and Shenlong Wang},
      year={2024},
      eprint={2409.01491},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.01491}, 
}
