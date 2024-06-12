export MODEL_NAME="runwayml/stable-diffusion-v1-5"

accelerate launch --mixed_precision="bf16" examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="data/3D" \
  --caption_column="Prompt" \
  --resolution=512 --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd15-3D-model-lora" \
  --validation_prompt="a cartoon ratty character standing in a room with a chair and a dog" --report_to="wandb"\
  --rank=4\
  --num_validation_images=1\
  --wandb_name="lora4"
