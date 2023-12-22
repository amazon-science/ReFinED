## Overview
End-to-end multilingual entity linking (MEL) is concerned with identifying multilingual entity mentions and their corresponding entity IDs in a knowledge base. Prior efforts assume that entity mentions are given and skip the entity mention detection step due to a lack of highquality multilingual training corpora. To overcome this limitation, we propose mReFinED, the first end-to-end MEL model. Additionally, we propose a bootstrapping mention detection framework that enhances the quality of training corpora. Our experimental results demonstrated that mReFinED outperformed the best existing work in the end-to-end MEL task while being 44 times faster.

### Hardware Requirements
mReFinED has a low hardware requirement. For fast inference speed, a GPU should be used, but this is not a strict requirement. 


#### mReFinED Paper
The mReFinED model architecture is described in the paper below (https://aclanthology.org/2023.findings-emnlp.1007):
```bibtex
@inproceedings{limkonchotiwat-etal-2023-mrefined,
    title = "m{R}e{F}in{ED}: An Efficient End-to-End Multilingual Entity Linking System",
    author = "Limkonchotiwat, Peerat  and
      Cheng, Weiwei  and
      Christodoulopoulos, Christos  and
      Saffari, Amir  and
      Lehmann, Jens",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.1007",
    doi = "10.18653/v1/2023.findings-emnlp.1007",
    pages = "15080--15089",
    abstract = "End-to-end multilingual entity linking (MEL) is concerned with identifying multilingual entity mentions and their corresponding entity IDs in a knowledge base. Existing works assumed that entity mentions were given and skipped the entity mention detection step due to a lack of high-quality multilingual training corpora. To overcome this limitation, we propose mReFinED, the first end-to-end multilingual entity linking. Additionally, we propose a bootstrapping mention detection framework that enhances the quality of training corpora. Our experimental results demonstrated that mReFinED outperformed the best existing work in the end-to-end MEL task while being 44 times faster.",
}
```
 
## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC 4.0 License.

## Contact us
If you have questions please open Github issues instead of sending us emails, as some of the listed email addresses are no longer active.
