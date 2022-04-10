# PICa
[An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA](https://arxiv.org/pdf/2109.05014.pdf)

by [Zhengyuan Yang](http://zhengyuan.info/), [Zhe Gan](https://zhegan27.github.io/), [Jianfeng Wang](https://scholar.google.com/citations?user=vJWEw_8AAAAJ&hl=en), [Xiaowei Hu](https://scholar.google.com/citations?user=Pj0TwxwAAAAJ&hl=en), [Yumao Lu](https://www.linkedin.com/in/yumao/), [Zicheng Liu](https://zicliu.wixsite.com/mysite), and [Lijuan Wang](https://www.microsoft.com/en-us/research/people/lijuanw/)

The 36th AAAI Conference on Artificial Intelligence (AAAI), 2022, Oral


### Introduction
Can GPT-3 benefit multimodal tasks? We provide an empirical study of GPT-3 for knowledge-based VQA, [named PICa](https://arxiv.org/pdf/2109.05014.pdf). We show that prompting GPT-3 via the use of image captions with only 16 examples surpasses supervised sota by an absolute +8.6 points on the OK-VQA dataset (from 39.4 to 48.0).

<p align="center">
  <img src="https://zyang-ur.github.io//pica/intro.jpg" width="75%"/>
</p>

### Citation

    @inproceedings{yang2021empirical,
      title={An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA},
      author={Yang, Zhengyuan and Gan, Zhe and Wang, Jianfeng and Hu, Xiaowei and Lu, Yumao and Liu, Zicheng and Wang, Lijuan},
      booktitle={AAAI},
      year={2022}
    }

### Prerequisites

* Obtain the public [OpenAI GPT-3 API key](https://openai.com/api/) and install the [API Python bindings](https://beta.openai.com/docs/api-reference/introduction).

## Installation

1. Clone the repository

    ```
    git clone https://github.com/microsoft/PICa.git
    ```

2. Prepare the data
The cached files for converted OKVQA data, predicted text representations, and similarity features are in the ``coco_annotations``, ``input_text``, and ``coco_clip_new`` folders, respectively.

### Running
3. We experimented with the older engine ``davinci`` instead of the current default ``text-davinci-001`` that is boosted for instruction tuning, see more discussion [here](https://beta.openai.com/docs/engines).
    ```
    python gpt3_api_okvqa.py --apikey xxx --output_path output

    ## for example
    python gpt3_api_okvqa.py --apikey xxx --output_path output --engine davinci --similarity_metric random --n_ensemble 1 --n_shot 16
    python gpt3_api_okvqa.py --apikey xxx --output_path output --engine davinci --similarity_metric imagequestion --n_ensemble 5 --n_shot 16
    ```

### Results
4. Outputs will be saved to ``format_answer`` and ``prompt_answer`` folders. ``format_answer`` is used for [final evaluation](https://github.com/GT-Vision-Lab/VQA), following the [vqav2 format](https://visualqa.org/evaluation.html). ``prompt_answer`` contains the input prompt for human interpretation. 

5. ``output_saved`` provides the cached predictions.