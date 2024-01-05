<!-- <h3 align="center"><img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150"></h3> -->

<h3 align="center">
Research Paper Artifacts
<br>
cuSZ-I: High-Fidelity Error-Bounded Lossy Compression for Scientific Data on GPUs
</h3>

<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
</p>

The research paper artifacts serve the reproducting purpose. The artifacts are developed by the paper authors, Jinyang Liu, Jiannan Tian, and Shixun Wu.


<h3 align="center">
build from source code
</h3>

- NVIDIA GPU with CUDA 11.3 onward
    <!-- - see detailed compatibility matrix below
    - Spack installation can work for 11.0 onward -->
- cmake 3.18 onward
- C++17 enabled compiler, GCC 9 onward

<b>To build cuSZ-I (cuSZ)</b>

```bash
# Example architectures (";" to separate multiple SM versions)
# A100: 80, A4000: 86
# Install to [/path/to/install/dir]

git clone https://github.com/szcompressor/cuSZ.git cusz-latest
cd cusz-latest && mkdir build && cd build

cmake .. \
    -DPSZ_BACKEND=cuda \
    -DPSZ_BUILD_EXAMPLES=on \
    -DCMAKE_CUDA_ARCHITECTURES="80;86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_COLOR_DIAGNOSTICS=on \
    -DCMAKE_INSTALL_PREFIX=[/path/to/install/dir]
make -j
make install
# `ctest` to perform testing
```

<b>To build cuZFP</b>

```bash
git clone https://github.com/LLNL/zfp.git
cd zfp && mkdir build && cd build
cmake .. \
    -DZFP_WITH_CUDA=on \
    -DCMAKE_CUDA_ARCHITECTURES="80;86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=[/path/to/install/dir]
make -j
make install
```

<b>To build cuSZp</b>

```bash
git clone https://github.com/szcompressor/cuSZp.git
cd cuSZp && mkdir build && cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="80;86" \
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=[/path/to/install/dir]
make -j
make install
```


<details>
<summary>
Detailed lookup: CUDA GPU architectures (SM version) and representative GPUs.
</summary>

NVIDIA CUDA architectures and names and representative GPUs are listed below. 
More details can be found at [CUDA GPUs](https://developer.nvidia.com/cuda-gpus).


| SM id  | arch.  | grade/segment         | GPU product example       |
| ------ | ------ | --------------------- | ------------------------- |
| 60     | Pascal | HPC/ML                | P100                      |
| 70     | Volta  | HPC/ML                | V100                      |
| 75     | Turing | consumer/professional | RTX 20?0, Quadro RTX ?000 |
| 80     | Ampere | HPC/ML                | A100                      |
| 86     | Ampere | consumer/professional | RTX 30?0, RTX A?000       |
| 89 `*` | Ada    | consumer/professional | RTX 40?0, RTX ?000        |
| 90 `*` | Hopper | HPC/ML                | H100                      |

`*` as of CUDA 11.8

</details>


<h3 align="center">
data source
</h3>

All mentioned data in the research paper but RTM data are available on 

- [SDRB](https://sdrbench.github.io): Miranda, Nyx, QMCPack, S3D
- [JHTDB](http://turbulence.pha.jhu.edu): Turbulence

<h3 align="center">
run
</h3>

Note that `cusz` has been changed for the artifacts by

- changing the binary name to `cuszi`, and 
- using `spline` as the default predictor (whereas `lorenzo` is the default for `cusz`)


```bash
# run cusz-interpolation using spline predictor
cuszi -t f32 -m r2r -e [ErrorBound] -i [/PATH/TO/DATA] -l [X]x[Y]x[Z] -z --report time
cuszi -i [/PATH/TO/DATA].cusza -x --report time --compare ${CESM}
```

```bash
# run cusz-lorenzo
cuszi -t f32 -m r2r -e [ErrorBound] -i [/PATH/TO/DATA] -l [X]x[Y]x[Z] -z --report time --predictor lorenzo
cuszi -i [/PATH/TO/DATA].cusza -x --report time --compare ${CESM}
```

```bash
# run FZ-GPU
./fz-gpu [/PATH/TO/DATA] [X] [Y] [Z] [ErrorBound]
```

```bash
# run cuSZp (draft)
./cuSZp_gpu_f32_api [/PATH/TO/DATA] REL [ErrorBound]
```

```bash
# run cuzfp (draft)
./zfp -f -i [/PATH/TO/DATA] -3 [Z] [Y] [X] -r [Rate] -x cuda
```

<h3 align="center">
acknowledgements
</h3>

This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants [CCF-1617488](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1617488), [CCF-1619253](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1619253), [OAC-2003709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2003709), [OAC-1948447/2034169](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2034169), and [OAC-2003624/2042084](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2042084).

![acknowledgement](https://user-images.githubusercontent.com/10354752/196348936-f0909251-1c2f-4c53-b599-08642dcc2089.png)
