## Efficient Unified Demosaicing for Bayer and Non-Bayer Patterned Image Sensors<br><sub>PyTorch implementation of the ICCV 2023 paper</sub>


[![project_page](https://img.shields.io/badge/-project%20page-green)](https://ignoww.github.io/KLAP_project/)
[![arXiv](https://img.shields.io/badge/arXiv-2211.16374-red)](https://arxiv.org/abs/2307.10667) 
 
> **Efficient Unified Demosaicing for Bayer and Non-Bayer Patterned Image Sensors**<br>
> Haechang Lee*, Dongwon Park*, [Wongi Jeong](https://ignoww.github.io)*, Kijeong Kim, Hyunwoo Je, Dongil Ryu, [Se Young Chun](https://icl.snu.ac.kr/pi) (*co-first) <br>
> **[ICCV 2023]** <br>
> 
> [ignoww.github.io/KLAP_project](https://ignoww.github.io/KLAP_project)
> 
>**Abstract**: <br>
As the physical size of recent CMOS image sensors (CIS) gets smaller, the latest mobile cameras adopt unique non-Bayer color filter array (CFA) patterns (e.g., Quad, Nona, QxQ), which consist of homogeneous color units with adjacent pixels. These non-Bayer CFAs are superior to conventional Bayer CFA thanks to their changeable pixel-bin sizes for different light conditions, but may introduce visual artifacts during demosaicing due to their inherent pixel pattern structures and sensor hardware characteristics. Previous demosaicing methods have primarily focused on Bayer CFA, necessitating distinct reconstruction methods for non-Bayer CIS with various CFA modes under different lighting conditions. In this work, we propose an efficient unified demosaicing method that can be applied to both conventional Bayer RAW and various non-Bayer CFAs' RAW data in different operation modes. Our Knowledge Learning-based demosaicing model for Adaptive Patterns, namely KLAP, utilizes CFA-adaptive filters for only 1% key filters in the network for each CFA, but still manages to effectively demosaic all the CFAs, yielding comparable performance to the large-scale models. Furthermore, by employing meta-learning during inference (KLAP-M), our model is able to eliminate unknown sensor-generic artifacts in real RAW data, effectively bridging the gap between synthetic images and real sensor RAW. Our KLAP and KLAP-M methods achieved state-of-the-art demosaicing performance in both synthetic and real RAW data of Bayer and non-Bayer CFAs.

## Data
The three pickle files in `data` are the samples after **r-CM**.

## weights
Download **pretrained** KLAP weight for meta-test. You can download `KLAP_epoch250_psnr41.364_ssim0.976` [here](https://drive.google.com/file/d/18Rg8ozcbqwHiuHcJWkSnm4aFKIWYuOaE/view?usp=sharing)

## Demo
```.bash
# Run demo_KLAP.sh

sh demo_KLAP.sh
```
The results are saved to `./experiments/"save-dir"`.
`--mode` in `demo_KLAP.sh` determines the CFA(Color Filter Array) pattern for mosaicing(Bayer, Quad, Nona, QxQ). Also `--lambda_n2s` and `--lambda_reg` are the hyper-parameters of Noise2Self loss and pixel-binning loss, respectively. More detailed information is in the paper.


## Citation

```
@article{lee2023efficient,
      title={Efficient Unified Demosaicing for Bayer and Non-Bayer Patterned Image Sensors},
      author={Lee, Haechang and Park, Dongwon and Jeong, Wongi and Kim, Kijeong and Je, Hyunwoo and Ryu, Dongil and Chun, Se Young},
      journal={arXiv preprint arXiv:2307.10667},
      year={2023}
    }
```
