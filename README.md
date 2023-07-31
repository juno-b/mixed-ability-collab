<a name="readme-top"></a>
<!-- PROJECT SHIELDS -->
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/juno-b/mixed-ability-collab">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Mixed Ability Collaboration</h3>

  <p align="center">
    This repository contains code designed to obtain and filter gaze data and send that to a website.
    The functions inside the tobiiLive.py file are the most recent versions, tobiiTest was used to develop the eye tracking functions.
    This code contains: data collection, filtering/processing, and visualization functions for the Tobii Pro Fusion eye tracker.
    It is able to calculate centroids live, contains a custom calibration function, and can write data to a csv file afterwards.
    This code contains a Python implementation of the Tobii I-VT Fixation Filter, which is a fixation classification algorithm.
    There is also a sample webpage, web.html, with Javascript designed to handle live data receiving with a Flask server.
    For more information, see the full description at the top of the tobiiLive file.

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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com)-->

Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `juno-b`, `mixed-ability-collab`, `twitter_handle`, `juno-bartsch-85594a235`, `email_client`, `email`, `Mixed Ability Collaboration`, `project_description`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

[![Python][Python.org]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.

### Prerequisites

This project is built and tested on a Tobii Pro Fusion eye tracker.

### Installation

1. Install the Tobii Pro SDK (tested with v 1.11) [https://developer.tobiipro.com/python/python-sdk-reference-guide.html](https://developer.tobiipro.com/python/python-sdk-reference-guide.html)
2. Install the necessary packages: matplotlib, numpy, Flask (for the server), pygame (for custom calibration).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Juno Bartsch - junobartsch@gmail.com

Project Link: [https://github.com/juno-b/mixed-ability-collab](https://github.com/juno-b/mixed-ability-collab)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This project would not have been possible without the support and contributions of [Yanzi (Veronica) Lin](https://github.com/yanziv), [Joon Jang](https://github.com/joonbugs), and [Andrew Begel](https://github.com/abegel).

Created at the Carnegie Mellon University [VariAbility Lab](https://github.com/cmu-variability) during the Summer 2023 REUSE program.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/juno-b/mixed-ability-collab.svg?style=for-the-badge
[contributors-url]: https://github.com/juno-b/mixed-ability-collab/graphs/contributors
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/juno-bartsch-85594a235
[product-screenshot]: images/screenshot.png
[Python.org]: https://img.shields.io/badge/python-3.10-gray?labelColor=3670a0&style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/downloads/release/python-31011/
