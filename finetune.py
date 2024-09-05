import os
import shutil
import csv
import uuid
import zipfile
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from prompts import describe_image
from llm_methods import generate_embedding, generate_description

from replicate_methods import create_replicate_model, start_training

SKIP_RESIZE_AND_DESCRIPTION = False
SKIP_MODEL_CREATION = False
SKIP_TRAINING = False

FILE_EXTENSIONS = [".jpg", ".png"]
REPLICATE_USER_NAME = "replicate_user_name"
REPLICATE_MODEL_NAME = "replicate_model_name"
MAX_IMAGE_SIZE = 1024
PREFIX = "In the style of TOK, " # used as prefix for the image descriptions
PREFIX_WORD = "TOK"
SOURCE_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "source_images")
TRAINING_PACK_DIR = os.path.join(os.path.dirname(__file__), "data", "training_pack")

def main():

    # greet the user
    print("Welcome to the flux dev finetuning script!")

    if not SKIP_RESIZE_AND_DESCRIPTION:

        print("We will now prepare the training pack.")

        # check that data/source_images exists and has at least 1 image. if not - show error and exit
        if not os.path.exists(SOURCE_IMAGES_DIR):
            print(f"Error: {SOURCE_IMAGES_DIR} does not exist. Please put some images there and try again.")
            return
        image_count = len([f for f in os.listdir(SOURCE_IMAGES_DIR) if any(f.endswith(ext) for ext in FILE_EXTENSIONS)])
        if image_count == 0:
            print(f"Error: {SOURCE_IMAGES_DIR} does not have any images. Please put some images there and try again.")
            return

        # calculate how many images are in the folder. we count images as .jpg and .png files only
        print(f"Found {image_count} images in {SOURCE_IMAGES_DIR}")

        # create a new folder data/training_pack. if it exists - warn the user and ask for the confirmation to overwrite
        if os.path.exists(TRAINING_PACK_DIR):
            print(f"Warning: {TRAINING_PACK_DIR} already exists. We will overwrite it.")
            overwrite = input("Do you want to continue? (y/n)")
            if overwrite != "y":
                print("Exiting...")
                return
            # empty the folder
            for file in os.listdir(TRAINING_PACK_DIR):
                os.remove(os.path.join(TRAINING_PACK_DIR, file))

        else:
            os.makedirs(TRAINING_PACK_DIR)

        image_files = [f for f in os.listdir(SOURCE_IMAGES_DIR) if any(f.endswith(ext) for ext in FILE_EXTENSIONS)]
        for image_file in tqdm(image_files, desc="Copying and renaming images"):
            if any(image_file.endswith(ext) for ext in FILE_EXTENSIONS):
                original_extension = os.path.splitext(image_file)[1]
                new_image_file = f"{uuid.uuid4()}{original_extension}"
                shutil.copy(os.path.join(SOURCE_IMAGES_DIR, image_file), os.path.join(TRAINING_PACK_DIR, new_image_file))

        print(f"We will now ensure that every image is at least {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}. This might take a while...")

        # now let's go image by image and make sure the longest side is 1024px. if it's not - we resize it
        for image_file in tqdm(os.listdir(TRAINING_PACK_DIR), desc="Resizing images"):
            if any(image_file.endswith(ext) for ext in FILE_EXTENSIONS):
                image_path = os.path.join(TRAINING_PACK_DIR, image_file)
                image = Image.open(image_path)
                width, height = image.size
                longest_side = max(width, height)

                if longest_side > MAX_IMAGE_SIZE:
                    scale_factor = MAX_IMAGE_SIZE / longest_side
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    image = image.resize((new_width, new_height), Image.LANCZOS)

                image.save(image_path)

        print("We will now generate descriptions for all images. This might take a while...")

        # now let's go image by image and generate a description for each image
        for image_file in tqdm(os.listdir(TRAINING_PACK_DIR), desc="Generating descriptions"):
            image_description = generate_description(os.path.join(TRAINING_PACK_DIR, image_file), describe_image)
            with open(os.path.join(TRAINING_PACK_DIR, f"{os.path.splitext(image_file)[0]}.txt"), "w") as f:
                f.write(PREFIX + image_description)

    if not SKIP_MODEL_CREATION:
        # now let's create a new model on Replicate
        print("We will now create a new model on Replicate.")
        try:
            create_replicate_model(REPLICATE_USER_NAME, REPLICATE_MODEL_NAME)
        except Exception as e:
            print(f"Error: {e}")
            return

    if not SKIP_TRAINING:

        # zip the training pack in preparation for upload
        print("We will now zip the training pack.")
        # Create a zip file of the training pack, excluding any existing zip files
        zip_path = os.path.join(TRAINING_PACK_DIR, "training_pack.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(TRAINING_PACK_DIR):
                for file in files:
                    if not file.endswith('.zip'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, TRAINING_PACK_DIR)
                        zipf.write(file_path, arcname)
        print(f"Training pack zipped successfully. Size: {os.path.getsize(zip_path) / (1024 * 1024):.2f} MB")

        print("We will now start the training on replicate")
        try:
            training = start_training(
                owner = REPLICATE_USER_NAME,
                model_name = REPLICATE_MODEL_NAME,
                training_pack_path = os.path.join(TRAINING_PACK_DIR, "training_pack.zip"),
                prefix = PREFIX_WORD
            )

        except Exception as e:
            print(f"Error: {e}")
            return

        print(f"Training started: {training.id}")
        print(f"You can track the training at https://replicate.com/p/{training.id}")

    # ask the user if they also want to create embeddings for all the image descriptions
    create_embeddings = input("Do you also want to create embeddings for all the image descriptions? (y/n)")
    if create_embeddings == "y":
        print("We will now create embeddings for all the image descriptions.")
        # create a new csv file

        with open(os.path.join(TRAINING_PACK_DIR, "embeddings.csv"), "w", newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["image_file", "description", "embedding"])
            # for each image file in the training pack, create an embedding and store it all in a .csv with image file name, description and embedding
            for image_file in tqdm(os.listdir(TRAINING_PACK_DIR), desc="Generating embeddings for each image description"):
                if any(image_file.endswith(ext) for ext in FILE_EXTENSIONS):
                    image_path = os.path.join(TRAINING_PACK_DIR, image_file)
                    with open(os.path.join(TRAINING_PACK_DIR, f"{os.path.splitext(image_file)[0]}.txt"), "r") as desc_file:
                        image_description = desc_file.read()
                    embedding = generate_embedding(image_description)
                    writer.writerow([image_file, image_description, embedding])

        print(f"Embeddings created successfully and saved in {TRAINING_PACK_DIR}/embeddings.csv")


    print("that's it! GLHF")

if __name__ == "__main__":
    main()