export MODEL_NAME="runwayml/stable-diffusion-v1-5"

accelerate launch --mixed_precision="bf16" examples/text_to_image/train_text_to_image.py \
  --wandb_name="full" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="data/coco_train" \
  --caption_column="Prompt" \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=1 --checkpointing_steps=500 \
  --learning_rate=2e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="/data/coco-lora" \
  --validation_prompt="A brown, black and white bird resting on a tree branch" --report_to="wandb" \
  --cache_dir="/data/cache"
