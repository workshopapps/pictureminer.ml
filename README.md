# pictureminer.ml

## **How to use**
---

## From prebuilt Image
to run this app from the prebuils image, execute the following command

```bash
$ docker run -p 8000:8000 geofferyj/caption-api:latest
```
This will pull the latest image from docker hub

## Building locally

To build locally do the following

1. install git-lfs
``` bash
$ sudo apt install git-lfs
```
2. Clone the repository
``` bash
$ git clone https://github.com/workshopapps/pictureminer.ml.git
```
3. initiallize git-lfs in cloned repo directory
``` bash
$ git lfs install
```
4. Pull large files with git-lfs
``` bash
$ git lfs pull
```
5. Build docker image
``` bash
$ docker build -t caption-api .
```
6. Run docker image 
``` bash
$ docker run -p 8000:8000 caption-api
```

## **Endpoints**
## Docs
---
http://localhost:8000/docs

## Caption Generator
http://localhost:8000/caption-generator
