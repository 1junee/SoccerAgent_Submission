---
license: cc-by-sa-4.0
---

# Dataset Card for SoccerWiki

<!-- Provide a quick summary of the dataset. -->

This repository contains the database for paper **"Multi-Agent System for Comprehensive Soccer Understanding"** in [ACM Multimidia 2025](https://acmmm2025.org/).

**SoccerWiki** is a large-scale multimodal soccer knowledge base. The dataset integrates rich domain knowledge about soccer players, teams, referees, and venues, which is used to facilitate knowledge-driven reasoning and decision-making in various soccer-related tasks.

This dataset was built using data from [Wikipedia](https://www.wikipedia.org/) and [Flashscore](https://www.flashscore.com/), and includes detailed attributes and images for each entity. It consists of 9,471 players, 266 teams, 202 referees, and 235 venues, making it one of the most comprehensive resources for soccer-related information. If you meet any problem while downloading this database, you can e-mail *jy_rao@sjtu.edu.cn*, that I will send you a Aliyun link for downloading it.

### Related Links

- [📄 Paper](https://arxiv.org/abs/2505.03735)  
- [🌐 WebPage](https://jyrao.github.io/SoccerAgent)  
- [💻 Github](https://github.com/jyrao/SoccerAgent)
- [🏆 Benchmark](https://huggingface.co/datasets/Homie0609/SoccerBench)  


## Dataset Details

### Key Features:
- **Players**: Information on 9,471 professional soccer players, including personal attributes, career data, and player images.
- **Teams**: Data about 266 soccer teams, including team statistics, historical performance, and team logos.
- **Referees**: 202 referees with detailed profiles and officiating history.
- **Venues**: Information about 235 soccer venues worldwide, including location, capacity, and images.

### Usage:
SoccerWiki is designed to support knowledge-driven AI applications in soccer, including but not limited to:
- **SoccerAgent**: A multi-agent system we proposed that leverages domain expertise from SoccerWiki to solve complex soccer-related problems.
- **Data-driven Research**: Facilitates studies in AI, data mining, and machine learning applications related to soccer.
- **Multimodal Analysis**: The dataset supports tasks that require both textual and visual reasoning, such as image captioning, multimodal retrieval, and knowledge graph reasoning.

This dataset provides an invaluable resource for researchers and developers working on soccer analytics, AI4Sports, and multimodal learning systems.


### Licensing Information

The data used in this project is sourced from Wikipedia https://www.wikipedia.org/ and Flashscore https://www.flashscore.com/. Please note the following:

* **Wikipedia**: The content from Wikipedia is licensed under the [Creative Commons Attribution-Share-Alike License 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.en). You are free to use, share, and adapt the content, provided that you attribute the source and share any derivative works under the same license.

* **Flashscore**: The data from Flashscore is used under their [Terms of Service](https://www.flashscore.com/terms/). Please refer to the specific terms provided by Flashscore for details on how the data may be used, including any restrictions or attribution requirements.

Ensure that you follow the licensing conditions of these sources when using or distributing this dataset.


## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

``````
└─ SoccerWiki
     ├─ data
     │   ├─ player
     │   │   ├─ player1.json
     │   │   ├─ player2.json
     │   │   └─ ...
     │   ├─ referee
     │   ├─ team
     │   └─ venue
     └─ pic
         ├─ player
         │   ├─ player1
         │   │   ├─ pic1.png
         │   │   ├─ pic2.png
         │   │   └─ ...
         │   ├─ player2
         │   │   ├─ pic1.png
         │   │   ├─ pic2.png
         │   │   └─ ...
         │   └─ ...
         ├─ referee
         ├─ team
         └─ venue

``````

Here is an example of the text data for a player:

```json
{
    "FULL_NAME": "Aaron Hunt",
    "UNICODE": "lls9wmxs",
    "PLAYER_URL": "https://wikipedia.org/wiki/Aaron_Hunt",
    "PLAYER_IMAGE_URL": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Aaron_Hunt_2018.jpg/150px-Aaron_Hunt_2018.jpg",
    "INFOBOX": {
        "Youth career": {
            "1993–1997": "VfLOker",
            ...
        },
        "Senior career*": {
            "Total": "    413    (82)",
            "2003–2007": "WerderBremenII    37    (9)",
            ...
        },
        "International career": {
            "2002": "GermanyU16    4    (1)",
            ...
        },
        "Personal information": {
            "Height": "1.83 m(6 ft0 in)[1]",
            "Position(s)": "Attackingmidfielder",
            "Date of birth": "(1986-09-04)4September1986(age 37)[1]",
            "Place of birth": "Goslar,WestGermany[1]"
        }
    },
    "CONTENT": {
        "Honours": "Werder BremenDFB-Pokal: 2008–09 ...",
        "References": {},
        "Club career": "Hunt was born in Goslar, Lower Saxony. After spending his first season at Werder Bremen in the reserves the year the first team achieved the double, he ...",
        "External links": "Aaron Hunt at WorldFootball.netAaron Hunt at kicker (in German)",
        "Career statistics": {},
        "International career": "Having a German father and an English mother, Hunt was eligible to play for both Germany or England..."
    },
    "IMAGES": [
        "https://upload.wikimedia.org/wikipedia/commons/1/19/Aaron_Hunt_2018.jpg",
        "https://upload.wikimedia.org/wikipedia/en/8/8a/OOjs_UI_icon_edit-ltr-progressive.svg"
    ],
    "SUMMARY": "Aaron Hunt is a German former professional footballer who ..."
}
```


## Citation 

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

      @inproceedings{rao2025soccceragent,
            title = {Multi-Agent System for Comprehensive Soccer Understanding},
            author = {Rao, Jiayuan and Li, Zifeng and Wu, Haoning and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
            booktitle = {ACM Multimedia 2025},
            year = {2025}
      }

## Dataset Card Contact

If you have any questions, please feel free to contact jy_rao@sjtu.edu.cn or zifengli@sjtu.edu.cn.