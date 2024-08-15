#!/bin/bash

# Must exit and fail to build if any command fails
set -eo pipefail
umask 002

# Use this layer to add nodes and models

APT_PACKAGES=(
    #"package-1"
    #"package-2"
)
# Packages are installed after nodes so we can fix them...
PIP_PACKAGES=(
    "opencv-python==4.7.0.72"
    "onnxruntime==1.17.1"
    "onnxruntime-gpu==1.17.1"
    "insightface==0.7.3"
)

NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/jags111/efficiency-nodes-comfyui"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/cubiq/ComfyUI_InstantID"
    "https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb"
    "https://github.com/storyicon/comfyui_segment_anything"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/kijai/ComfyUI-Marigold"
    "https://github.com/Amorano/Jovimetrix"
)

CHECKPOINT_MODELS=(
     "https://civitai.com/api/download/models/354657"
     "https://civitai.com/api/download/models/471120?token=dd2e86188faf31ecef4ba9fb273de0b6"
    #"https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    #"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
    #"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    #"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
    # "https://huggingface.co/stabilityai/stable-cascade/resolve/main/comfyui_checkpoints/stable_cascade_stage_c.safetensors"
)

UNET_MODELS=(

)

LORA_MODELS=(
    #"https://civitai.com/api/download/models/16576"
    # "https://civitai.com/api/download/models/177940"
)

VAE_MODELS=(
    #"https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.safetensors"
    #"https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    #"https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
)

ESRGAN_MODELS=(
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
    "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
    "https://huggingface.co/Akumetsu971/SD_Anime_Futuristic_Armor/resolve/main/4x_NMKD-Siax_200k.pth"
)

CONTROLNET_MODELS=(
    # "https://huggingface.co/stabilityai/stable-cascade/resolve/main/controlnet/canny.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_depth-fp16.safetensors"
    #"https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_depth_fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_hed-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_mlsd-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_normal-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_scribble-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_seg-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_color-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_depth-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_keypose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_seg-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_sketch-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_style-fp16.safetensors"
)
### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function build_extra_start() {
    build_extra_get_apt_packages
    build_extra_get_nodes
    build_extra_get_pip_packages
    build_extra_get_models \
        "/opt/storage/stable_diffusion/models/ckpt" \
        "${CHECKPOINT_MODELS[@]}"
    build_extra_get_models \
        "/opt/storage/stable_diffusion/models/unet" \
        "${UNET_MODELS[@]}"
    build_extra_get_models \
        "/opt/storage/stable_diffusion/models/lora" \
        "${LORA_MODELS[@]}"
    build_extra_get_models \
        "/opt/storage/stable_diffusion/models/controlnet" \
        "${CONTROLNET_MODELS[@]}"
    build_extra_get_models \
        "/opt/storage/stable_diffusion/models/vae" \
        "${VAE_MODELS[@]}"
    build_extra_get_models \
        "/opt/storage/stable_diffusion/models/esrgan" \
        "${ESRGAN_MODELS[@]}"

    BASE_DIR="/opt/storage/stable_diffusion/models"

    # Download clip vision
    CLIP_VISION_DIR="$BASE_DIR/clip_vision"
    mkdir -p "$CLIP_VISION_DIR"
    wget -nc --content-disposition "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" -O "$CLIP_VISION_DIR/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
     
    # Download ipadapter
    IPADAPTER_DIR="$BASE_DIR/ipadapter"
    mkdir -p "$IPADAPTER_DIR"
    wget -nc --content-disposition "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" -P "$IPADAPTER_DIR"
    wget -nc --content-disposition "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sdxl.safetensors" -P "$IPADAPTER_DIR"

    # Download models for instandid
    INSTANTID_DIR="$BASE_DIR/instantid"
    mkdir -p "$INSTANTID_DIR"
    wget -nc --content-disposition  -P "$INSTANTID_DIR" "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin"

    
    INSIGHTFACE_DIR="$BASE_DIR/insightface/models"
    mkdir -p "$INSIGHTFACE_DIR"
    wget -nc --content-disposition  -P "$INSIGHTFACE_DIR" "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip"
    unzip -q -o "$INSIGHTFACE_DIR/antelopev2.zip" -d "$INSIGHTFACE_DIR"
     
    # Download grounding dino model
    GROUNDING_DINO_DIR="$BASE_DIR/grounding-dino"
    mkdir -p "$GROUNDING_DINO_DIR"
    wget -nc --content-disposition  -P "$GROUNDING_DINO_DIR" "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py"
    wget -nc --content-disposition  -P "$GROUNDING_DINO_DIR" "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"

    # Download models for bert base uncased
    BERT_BASE_UNCASED_DIR="/opt/storage/stable_diffusion/models/bert-base-uncased"
    mkdir -p "$BERT_BASE_UNCASED_DIR"
    wget -nc --content-disposition  -P "$BERT_BASE_UNCASED_DIR" "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors"
    wget -nc --content-disposition  -P "$BERT_BASE_UNCASED_DIR" "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt"
    wget -nc --content-disposition  -P "$BERT_BASE_UNCASED_DIR" "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json"
    wget -nc --content-disposition  -P "$BERT_BASE_UNCASED_DIR" "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json"
    wget -nc --content-disposition  -P "$BERT_BASE_UNCASED_DIR" "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json"

    # Download models for segment anything
    SAM_DIR="$BASE_DIR/sams"
    mkdir -p "$SAM_DIR"
    wget -nc --content-disposition  -P "$SAM_DIR" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


    # Download controlnet models
    CONTROLNET_DIR="$BASE_DIR/controlnet"
    mkdir -p "$CONTROLNET_DIR"
    wget -nc --content-disposition  "https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors" -O "$CONTROLNET_DIR/t2iadapter_lineart-fp16.safetensors"
    wget -nc --content-disposition  "https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors" -O "$CONTROLNET_DIR/t2iadapter_skectch-fp16.safetensors"
    wget -nc --content-disposition  "https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors" -O "$CONTROLNET_DIR/t2iadapter_canny-fp16.safetensors"
    wget -nc --content-disposition  "https://huggingface.co/TencentARC/t2i-adapter-depth-zoe-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors" -O "$CONTROLNET_DIR/t2iadapter_depth-zoe-fp16.safetensors"
    wget -nc --content-disposition  "https://huggingface.co/TencentARC/t2i-adapter-depth-midas-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors" -O "$CONTROLNET_DIR/t2iadapter_depth-midas-fp16.safetensors"
    wget -nc --content-disposition  "https://huggingface.co/TencentARC/t2i-adapter-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors" -O "$CONTROLNET_DIR/t2iadapter_openpose-fp16.safetensors"
    wget -nc --content-disposition  "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors" -O "$CONTROLNET_DIR/instantid-fp16.safetensors"
    wget -nc --content-disposition  "https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic/resolve/main/TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors" -O "$CONTROLNET_DIR/TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors"

    # Download animatediff_models
    ANIMATEDIFF_DIR="$BASE_DIR/animatediff_models"
    mkdir -p "$ANIMATEDIFF_DIR"
    wget -nc --content-disposition "https://huggingface.co/hotshotco/Hotshot-XL/resolve/main/hsxl_temporal_layers.safetensors" -O "$ANIMATEDIFF_DIR/hsxl_temporal_layers.safetensors"

    # Marigold
    cd "$BASE_DIR/diffusers"
    git clone https://huggingface.co/Bingxin/Marigold

    cd /opt/ComfyUI
    source "$COMFYUI_VENV/bin/activate"
    LD_PRELOAD=libtcmalloc.so python main.py \
        --cpu \
        --listen 127.0.0.1 \
        --port 11404 \
        --disable-auto-launch \
        --quick-test-for-ci
    deactivate
}

function build_extra_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/opt/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                    "$COMFYUI_VENV_PIP" install --no-cache-dir \
                        -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                "$COMFYUI_VENV_PIP" install --no-cache-dir \
                    -r "${requirements}"
            fi
        fi
    done
}

function build_extra_get_apt_packages() {
    if [ ${#APT_PACKAGES[@]} -gt 0 ]; then
        $APT_INSTALL ${APT_PACKAGES[*]}
    fi
}
function build_extra_get_pip_packages() {
    if [ ${#PIP_PACKAGES[@]} -gt 0 ]; then
        "$COMFYUI_VENV_PIP" install --no-cache-dir \
            ${PIP_PACKAGES[*]}
    fi
}

function build_extra_get_models() {
    if [[ -n $2 ]]; then
        dir="$1"
        mkdir -p "$dir"
        shift
        arr=("$@")
        
        printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
        for url in "${arr[@]}"; do
            printf "Downloading: %s\n" "${url}"
            build_extra_download "${url}" "${dir}"
            printf "\n"
        done
    fi
}

# Download from $1 URL to $2 file path
function build_extra_download() {
    wget -nc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
}

umask 002

build_extra_start
fix-permissions.sh -o container