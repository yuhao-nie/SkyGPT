# SkyGPT: Probabilistic Short-term Solar Forecasting Using Synthetic Sky Videos from Physics-constrained VideoGPT

<!--

(Include an eye-catching demo animation here, want to show both stochastic video prediction and probabilistic PV output prediction)


---
-->

[**Pre-print**](https://arxiv.org/abs/2306.11682)

The variability of solar photovoltaic (PV) output, driven by rapidly changing cloud dynamics, hinders the transition to reliable renewable energy systems. Although sky image-based methods using deep learning achieves the state-of-the-art in forecasting solar power, two major gaps exist:

1. Existing methods often struggle to accurately capture cloud dynamics. Temporal lags in predictions are often observed when PV generation fluctuates during short-term cloudy events.
2. There is commonly lack of uncertainty quantification for solar power predictions, which however is critical to risk management in renewable-heavy grids.

Information on future sky conditions, especially cloud coverage, is beneficial for PV output forecasting. With the recent advances in generative artificial intelligence, synthesis of possible images of the future sky has potential for aiding in forecasts of solar PV power generation. 

Here, we introduce *SkyGPT*, a physics-constrained stochastic video prediction model, to generate plausible future sky videos with diverse cloud motion patterns based on past sky image sequences. We demonstrate the potential of using the synthetic future sky images from *SkyGPT* for a 15-minute-ahead probabilistic PV output forecasting task by coupling it with a PV output prediction model (a modified version of U-Net, see PV Output Prediction Section for more details) using real-world power generation data from a 30-kW rooftop PV system. An illustration of the framework is presented in the figure below.

![solar_forecasting_framework](/figures/proposed_forecasting_system_v3.png)
<p align=justify>
Figure 1: Proposed probabilistic solar forecasting framework. Historical sky images from the past 15 minutes with time stamps 2 minutes apart from each other are used as input to predict the next 15 minutes' future sky scenarios with the same temporal resolution. Sky image frames are generated in an iterative fashion and only the last predicted frames (at time $t+15$) are used for PV output prediction. The same PV output prediction model is employed to map all the predicted images at $t+15$ to PV generation. A distribution of PV output prediction is obtained based on the collection of PV output predictions.
</p>

---

## Code Base and Dependencies
All the codes are writen in Python 3.6.1. The stochastic video prediction model *SkyGPT* is implemented using deep learning framework Pytorch 1.8.1, and the PV output prediction model is implemented using deep learning framework TensorFlow 2.4.1. Both deep learning models are trained on GPU cluster, with NVIDIA TESLA A100 40GB card. All dependencies are listed in the file `requirements.txt`. 

A list of all code files in the `/codes` folder can be found in the table below. It should be noted that for the video prediction models, we only provided the codes for ConvLSTM, PhyDNet+GAN and SkyGPT here. The codes for ConvLSTM and PhyDNet+GAN models are adapted from the PhyDNet and the codes for SkyGPT are modified based on the code base of VideoGPT. For the codes associated with PhyDNet and VideoGPT, please refer to their original GitHub repositories: https://github.com/vincent-leguen/PhyDNet and https://github.com/wilson1yan/VideoGPT. We implemented these two models with their default training setups using our own dataset.

| File | Description |
| ------------- | ------------- |
|`video_prediction/` | |
|&nbsp; `ConvLSTM/ConvLSTM.ipynb` |Jupyter Notebook used to train and validate the ConvLSTM model for sky video prediction.|
|&nbsp; `ConvLSTM/models.py` |Python file contains functions used in ConvLSTM_sky_image_dataset.ipynb notebook.|
|&nbsp; `PhyDNetGAN/PhyDNetGAN.ipynb` |Jupyter Notebook used to train and validate the PhyDNet+GAN model for sky video prediction.|
|&nbsp; `PhyDNetGAN/models_v2.py` |Python file contains functions used in ConvLSTM_sky_image_dataset.ipynb notebook.|
|&nbsp; `PhyDNetGAN/constrain_moments.py` |Python file contains the moment loss functnction used in PhyCell training to enforce the convolutions for apporximating spatial derivatives (see [PhyDNet paper](https://arxiv.org/abs/2003.01460) for details).|
|&nbsp; `SkyGPT/model/attention.py` |Python file contains attention modules for the transformer part in SkyGPT.|
|&nbsp; `SkyGPT/model/constrain_moments.py` |Python file contains moment loss function used in PhyCell to enforce the convolutions for approximating spatial derivatives, same as that used in PhyDNet. We incoporate PhyCell into the transformer part of the SkyGPT architecture to model physical dynamics.|
|&nbsp; `SkyGPT/model/gpt.py` |Python file contains functions for the transformer part of SkyGPT.|
|&nbsp; `SkyGPT/model/resnet.py` |Python file contains resnet module as encoder for SkyGPT.|
|&nbsp; `SkyGPT/model/utils.py` |Python file contains helper functions used in SkyGPT.|
|&nbsp; `SkyGPT/model/vqvae.py` |Python file contains the VQ-VAE part of SkyGPT.|
|&nbsp; `SkyGPT/script/training/training_vqvae.py` |Python file for training VQ-VAE.|
|&nbsp; `SkyGPT/script/training/training_transformer.py` |Python file for training transformer.|
|&nbsp; `SkyGPT/script/sampling/reformat_input.py` |Python file for reformating the input data for SkyGPT.|
|&nbsp; `SkyGPT/script/sampling/sample_gen.py` |Python file for generating samples for SkyGPT.|
|&nbsp; `eval/quan_eval_det_models.ipynb` |Jupyter Notebook used to quantatively evaluate the performance of deterministic video prediction models ConvLSTM, PhyDNet and PhyDNet+GAN using metrics including MAE, MSE and VGG Cosine Similarity.|
|&nbsp; `eval/quan_eval_VideoGPT.ipynb` |Jupyter Notebook used to quantatively evaluate the performance of stochastic video prediction model VideoGPT for using metrics including MAE, MSE and VGG Cosine Similarity.|
|&nbsp; `eval/quan_eval_SkyGPT.ipynb` |Jupyter Notebook used to quantatively evaluate the performance of stochastic video prediction model SkyGPT for using metrics including MAE, MSE and VGG Cosine Similarity.|
|&nbsp; `eval/comp_all_models.ipynb` |Jupyter Notebook used to compare the performance of all video prediction models based on the metrics including MAE, MSE and VGG Cosine Similarity.|
|`pv_output_prediction/` | |
|&nbsp; `SUNSET_PV_forecast.ipynb` |Jupyter Notebook used to train, validate and test SUNSET, which is a AlexNet-like CNN model, for 15-minute-ahead PV output forecast using the past 15-minute sky images and PV output record as model input.|
|&nbsp; `UNet_sky_image_PV_mapping.ipynb` |Jupyter Notebook used to train validate and test a modified version of UNet, which essentially learns a map from sky images to PV output value. Once the future sky images are generated by video prediction models, they can be fed to UNet to predict 15-minute-ahead PV power output.|



## Dataset
We leverage an in-house dataset ($\mathscr{D}$) with 334,038 aligned pairs of sky images ($\mathcal{I}$) and PV power generation ($\mathcal{P}$) records, $\mathscr{D} = \{(\mathcal{I}_i, \mathcal{P}_i) \mid i\in \mathbb{Z}: 1\leq i\leq 334\mathrm{,}038\}$, for the experiments in this study. Please check out our [paper](https://arxiv.org/abs/2306.11682) for details about the dataset and data processing steps.

The data used in this study are stored in [Google Drive](https://drive.google.com/drive/folders/1J2I-Aj70mbvuocwHCo-PYurBhfCCZUGh?usp=sharing) and can be accessed for free. A list of data files  is shown in the table below.
| File | Description |
| ------------- | ------------- |
|`video_prediction_dataset.hdf5` | A file-directory like structure consisting of two groups: "trainval" and "test", for storing model development set and test set, respectively, for video prediction and PV output prediction. Each group contains four types of data: "images_log", "images_pred", "pv_log" and "pv_pred", which stores the sky images and PV generation data from 2017 March to 2019 October in Python NumPy array format. "images_log" contains data of historical images from time $t-15$ to $t$ with 1-minute interval (i.e., 16 images), intended to be used as model input, with a shape of (N, 16, 64, 64, 3), where N stands for the number of samples, 16 represents the temporal dimension, and images are downsized to $64\times64$ with RGB channels. "images_pred" contains data of future images from time $t+1$ to $t+15$ with 1-minute interval (i.e., 15 images), to be used as the training target of video prediction model, with a shape of (N, 15, 64, 64, 3). Similarly, "pv_log" and "pv_pred" contains historical and future PV power generation data, respectively, corresponding to "images_log" and "images_pred" data, with a shape of (N, 16) and (N, 15), intended to be used in training PV output prediction models.|
| `times_curr_trainval.npy` | Python NumPy array of time stamps corresponding to time $t$ of the development set in video_prediction_dataset.hdf5 file.  |
| `times_curr_test.npy` | Python NumPy array of time stamps corresponding to time $t$ of the test set in video_prediction_dataset.hdf5 file.  |
| `test_set_2019nov_dec.hdf5` |  A file-directory like structure consisting of only one group: "test", for storing additional test data (5 cloudy days) from 2019 November to December, which is outside the timeframe of the data in video_prediction_dataset.hdf5 to test how well the model extrapolates. Similar to video_prediction_dataset.hdf5, it contains four types of data: "images_log", "images_pred", "pv_log" and "pv_pred" in Python Numpy array format.|
| `times_curr_test_2019nov_dec.npy` |  Python NumPy array of time stamps corresponding to time $t$ of the test set in test_set_2019nov_dec.hdf5 file. |

Note: This study was conducted before the official release of our curated dataset [SKIPP'D](https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset) [[5](#5)], which is more organized and has a number of updates from the dataset we used here. We encourage the readers to examine the SKIPP'D dataset.

## Stochastic Sky Video prediction
As a first step, we train video prediction models to generate future sky images based on past sky image sequences. We name our proposed stochastic sky video prediction model *SkyGPT*, which is inspired by two emerging video prediction models VideoGPT [[1](#1)] and PhyDNet [[2](#2)]. The SkyGPT follows the general structure of VideoGPT, which consists of two main parts, a vector quantized variational auto-encoder (VQ-VAE) [[3](#3)] and an image transformer [[4](#4)]. The VQ-VAE encompasses an encoder-decoder architecture similar to classical VAEs, but it learns a discrete latent representation of input data instead of a continuous one. The image transformer, as a prior network, is used to model the latent tokens in an auto-regressive fashion, where new predictions are made by feeding back the predictions from previous steps. To enhance the modeling of cloud motion, we incorporate prior physical knowledge into the transformer by adapting a PDE-constrained module called PhyCell from the PhyDNet [[2](#2)] for latent modeling. We call this entire architecture a Phy-transformer (in short of physics-informed transformer) to distinguish it from the transformer component within the architecture. 

<p align="center">
<img src="figures/SkyGPT_for_future_sky_image_prediction_v2.png" alt="skygpt" width="70%" height="auto">
</p>
<p align=justify>
Figure 2: SkyGPT for future sky image prediction. The prediction is disentangled in the encoding space by the PhyCell and Transformer. For visualization purposes, the next step encodings predicted by PhyCell and Transformer are decoded, which shows that PhyCell captures the physical pattern of the motion, while Transformer is responsible for filling in the prediction with fine-grained details.
</p>

## PV Output Prediction
As a second step, we train PV output prediction model that learns a mapping from the sky image to concurrent PV power output. Such a mapping can be trained on historical real-world images and then applied to our generated future sky images. An analogy one can think of is the computer vision task of estimating the age of people based on their facial images.

The PV output predictor is based on U-Net [[5](#5)], which has an encoder-bottleneck-decoder architecture and is commonly used in various image segmentation tasks. For the PV output prediction task, a few modifications were made to the architecture of U-Net, including (1) changing the output of the original U-Net to generate a regression result instead of a segmentation map, (2) using residual block for the bottleneck part instead of the classical Convolution-BatchNorm-ReLU structure to ease the network training, (3) pruning the architecture by reducing the number of convolution layers.

<p align="center">
<img src="figures/UNet.png" alt="unet" width="80%" height="auto">
<p align=center>
Figure 3: Modified U-Net architecture for PV output prediction.
</p>

## Results
### Future Sky Images Generated by Video Prediction Models
Here, we demonstrate two examples of predicted videos by SkyGPT compared to benchmark models. The two examples reflect two different cloud dynamics: (a) the sky changing from partly cloudy to overcast condition and (b) the sky changing from partly cloudy to clear sky condition. All models start with the same context frames as input, and *SkyGPT* shows noticeably more accurate and diverse prediction compared with the deteriministic models. More detailed results can be found in our [paper](https://arxiv.org/abs/2306.11682).

<p align=center>
(a) The sky condition changing from partly cloudy to overcast
</p>

![video_pred_demo_1](/figures/video_pred_demo_1.gif)

<p align=center>
(b) The sky condition changing from partly cloudy to clear sky
</p>

![video_pred_demo_2](/figures/video_pred_demo_2.gif)

### Synthetic Future Sky Images for Probabilistic Solar Forecasting

We then demonstrate using the predicted future sky images by *SkyGPT* as input for U-Net model for 15-minute-ahead PV output prediction compared with an end-to-end deep solar forecasting model [SUNSET](https://github.com/YuchiSun/SUNSET). The animations below show the predictions of these two models on two cloudy days with increasing level of variations in PV generation. The proposed *SkyGPT*->U-Net framework achieves superior prediction reliability and sharpness over SUNSET. More detailed results can be found in our [paper](https://arxiv.org/abs/2306.11682).

![pred_curve_demo_1](/figures/pred_curve_demo_1.gif)
![pred_curve_demo_2](/figures/pred_curve_demo_2.gif)

## Reference
<a id="1">[1]</a> 
Yan, W., Zhang, Y., Abbeel, P. and Srinivas, A., 2021. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157.

<a id="2">[2]</a>
Guen, V.L. and Thome, N., 2020. Disentangling physical dynamics from unknown factors for unsupervised video prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 11474-11484).

<a id="3">[3]</a>
Van Den Oord, A. and Vinyals, O., 2017. Neural discrete representation learning. Advances in neural information processing systems, 30.

<a id="4">[4]</a>
Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D. and Sutskever, I., 2020, November. Generative pretraining from pixels. In International conference on machine learning (pp. 1691-1703). PMLR.

<a id="5">[5]</a>
Nie, Y., Li, X., Scott, A., Sun, Y., Venugopal, V. and Brandt, A., 2023. SKIPP’D: A SKy Images and Photovoltaic Power Generation Dataset for short-term solar forecasting. Solar Energy, 255, pp.171-179.
