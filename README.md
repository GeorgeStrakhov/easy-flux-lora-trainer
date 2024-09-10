# simple script for easy flux finetuning from a folder of images. uses replicate for the cloud gpu goodness.

0. populate .env file: add your openai and replicate keys. openai used for captioning.
1. `pip install -r requirements.txt`
2. put all the images inside `data/source_images`
3. adjust constants at the top of `finetune.py` (your replicate details)
4. run `python finetune.py`
5. wait for the script to finish and it will return the the training url
6. optionally create embeddings for all the image descriptions and store them in a .csv - so that later you can do semantic searches over your image library


## Behind the scenes the script will:

0. ask the user for details e.g. what to call the model
1. create a new folder data/training_pack and copy + convert all images and rename to uuid.jpg format
2. if need be - downscale images to 1024x1024 (max)
3. run each image through gpt4-o-mini to generate a description
4. save all descriptions as uuid.txt and put it into the same folder, optionally creates embeddings and adds to the csv
5. .zip the folder
6. create a new model on replicate
7. create a new training job on replicate - and give the user the url to check on the training
