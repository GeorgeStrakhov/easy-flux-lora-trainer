import replicate

def create_replicate_model(owner, name, visibility="private", hardware="gpu-t4", description="A fine-tuned FLUX.1 model"):
    model = replicate.models.create(
        owner=owner,
        name=name,
        visibility=visibility,
        hardware=hardware,
        description=description
    )

    print(f"Model created: {model.name}")
    print(f"Model URL: https://replicate.com/{model.owner}/{model.name}")

    return model


def start_training(owner, model_name, training_pack_path, prefix):
    # Now use this model as the destination for your training
    destination=f"{owner}/{model_name}"
    print(f"Starting training for {destination}")
    training = replicate.trainings.create(
        version="ostris/flux-dev-lora-trainer:4ffd32160efd92e956d39c5338a9b8fbafca58e03f791f6d8011f3e20e8ea6fa",
        input={
            "input_images": open(training_pack_path, "rb"),
            "steps": 1000,
            "lora_rank": 24,
            "optimizer": "adamw8bit",
            "batch_size": 1,
            "resolution": "512,768,1024",
            "autocaption": False,
            "trigger_word": prefix,
            "learning_rate": 0.0004,
            "caption_dropout_rate": 0.05,
        },
        destination=destination
    )

    return training