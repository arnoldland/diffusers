MODEL_NAME="runwayml/stable-diffusion-v1-5"
rank=$1
echo "Rank: $rank"
dataset="coco_train"
wandb_name="lora_${rank}_coco"
output_dir="result/lora_${rank}_coco"
image_output_dir="/data/coco-lora-${rank}"
export CURL_CA_BUNDLE=''
export REQUESTS_CA_BUNDLE=''
accelerate launch --num_processes 4 --mixed_precision="bf16" --main_process_port 19292 examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="data/$dataset" \
  --caption_column="Prompt" \
  --mixed_precision="bf16" \
  --resolution=512 --random_flip \
  --center_crop \
  --train_batch_size=8 \
  --num_train_epochs=1 \
  --checkpointing_steps=5000 \
  --learning_rate=5e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=${output_dir} \
  --validation_prompt="A brown, black and white bird resting on a tree branch" \
  --report_to="wandb" \
  --rank=${rank} \
  --num_validation_images=4 \
  --wandb_name=$wandb_name
accelerate launch --num_processes 4 inference.py ${output_dir} ${image_output_dir} --is_lora True
echo ${image_output_dir} >> output.txt
python -m pytorch_fid data/coco_val/val2017/ ${image_output_dir} >> output.txt

