<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![PyPI][pypi-shield]][pypi-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/hypernetwork-research-group/hypernegative">
    <img src="https://github.com/hypernetwork-research-group/hypernegative/raw/main/assets/images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">HYPERNEGATIVE</h3>

  <p align="center">
    A Python library for the evaluation of Hyperlink Prediction algorithms
    <br />
    <a href="https://github.com/hypernetwork-research-group/hypernegative"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/hypernetwork-research-group/hypernegative">View Demo</a>
    &middot;
    <a href="https://github.com/hypernetwork-research-group/hypernegative/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/hypernetwork-research-group/hypernegative/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://www.isislab.it/)

***Hypernegative*** is a Python library designed for the evaluation of Hyperlink Prediction (HLP) models.
It provides a unified interface for all components of an evaluation pipeline, ensuring consistency, modularity, and ease of use.

The library is structured as a modular and reusable framework, with a strong focus on reproducibility in both Hyperlink Prediction (HLP) and Negative Sampling (NS) methods.

Originally developed as a Bachelor’s thesis project in Computer Science at the University of Salerno, Hypernegative is intended to evolve into a research and experimentation tool in the domains of HLP and NS.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python.js]][Python-url]
* [![PyTorch][Pytorch.js]][Pytorch-url]
* [![Pytorch_geometric][PT_geom.js]][PT_geom-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
Follow these steps to set up the project locally.  

### Installation  

Hypernegative supports **Python 3.9 to 3.13**.  

You can install Hypernegative, which requires **PyTorch** and **PyTorch Geometric (PyG)**, by running:  

You can install and use Hypernergative wich require the library PyTorch and PyG. For this, simply run
   ```sh
   pip install hypernegative
   ```

<!-- USAGE EXAMPLES -->
## Usage
You can either use Hypernegative as a **Python library** or through the **CLI**.  

### Python example  

```python
from hypernegative.hyperlink_prediction.datasets import IMDBHypergraphDataset
from hypernegative.hyperlink_prediction.loader import DatasetLoader

dataset = IMDBHypergraphDataset()
loader = DatasetLoader(
    dataset,
    "MotifHypergraphNegativeSampler", 
    dataset._data.num_nodes,
    batch_size=4000, 
    shuffle=True, 
    drop_last=True
)
```
### CLI example

Show available options
```sh
imdb_pipeline --help
```
Run a pipeline with a specific dataset, negative sampling strategy, and HLP method:
```sh
imdb_pipeline --dataset_name COURSERA --negative_sampling MotifHypergraphNegativeSampler --hlp_method CommonNeighbors
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Giovanni Semioli - g.semioli1@studenti.unisa.it

Project Link: [https://github.com/hypernetwork-research-group/hypernegative](https://github.com/hypernetwork-research-group/hypernegative)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [ISISLab](https://www.isislab.it/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/hypernetwork-research-group/hypernegative.svg?style=for-the-badge
[contributors-url]: https://github.com/hypernetwork-research-group/hypernegative/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hypernetwork-research-group/hypernegative.svg?style=for-the-badge
[forks-url]: https://github.com/hypernetwork-research-group/hypernegative/network/members
[stars-shield]: https://img.shields.io/github/stars/hypernetwork-research-group/hypernegative.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/hypernetwork-research-group/hypernegative.svg?style=for-the-badge
[issues-url]: https://github.com/hypernetwork-research-group/hypernegative/issues
[license-shield]: https://img.shields.io/github/license/hypernetwork-research-group/hypernegative.svg?style=for-the-badge
[license-url]: https://github.com/hypernetwork-research-group/hypernegative/blob/master/LICENSE.txt
[product-screenshot]: https://github.com/hypernetwork-research-group/hypernegative/raw/main/assets/images/logo_isis.png 
[Python.js]: https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B
[Python-url]: https://www.python.org/
[Pytorch.js]: https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=FFFFFF
[Pytorch-url]: https://pytorch.org/
[PT_geom.js]: https://img.shields.io/badge/PyG-502DA7?style=for-the-badge&logo=https%3A%2F%2Fsimpleicons.org%2F%3Fmodal%3Dicon%26q%3Dpyg&logoColor=FFFFFF
[PT_geom-url]: https://pytorch-geometric.readthedocs.io/en/latest/
[pypi-shield]: https://img.shields.io/pypi/v/hypernegative?style=for-the-badge
[pypi-url]: https://pypi.org/project/hypernegative/