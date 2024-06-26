MODEL_NAME="runwayml/stable-diffusion-v1-5"
dataset=$1
rank=$2
output_dir="result/lora_${rank}_${dataset}"
wandb_name="lora_${rank}_${dataset}"
accelerate launch --mixed_precision="bf16" examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="data/$dataset" \
  --caption_column="Prompt" \
  --resolution=512 --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=${output_dir} \
  --validation_prompt="a cartoon ratty character standing in a room with a chair and a dog" --report_to="wandb" \
  --rank=${rank} \
  --num_validation_images=4 \
  --wandb_name=$wandb_name \
