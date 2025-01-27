import boto3
from botocore.config import Config
import base64
import time
import os,sys
import json
import string
import random
import re
from PIL import Image
import io
from config import DEFAULT_BUCKET,SYSTEM_CANVAS,SYSTEM_IMAGE_TEXT,MODEL_OPTIONS,SYSTEM_TRANSLATE

# Initialize AWS clients
session = boto3.session.Session(region_name='us-east-1')
client = session.client(service_name='bedrock-runtime', 
                       config=Config(connect_timeout=1000, read_timeout=1000))
bedrock_runtime = session.client("bedrock-runtime")

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MAX_PROMPT_LENGTH = 512


def random_string_name(length=12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def load_system_prompts(guideline_path):
    with open(guideline_path, "rb") as file:
        doc_bytes = file.read()
    return doc_bytes

def parse_prompt(text: str, pattern: str = r"<prompt>(.*?)</prompt>") -> str:
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError(f"No match found for pattern: {pattern}")

def png_to_rgb_base64(image_path):
    try:
        with Image.open(image_path) as img:
            # 转换为RGB模式
            rgb_img = img.convert('RGB')
            
            buffer = io.BytesIO()
            rgb_img.save(buffer, format='PNG')
            image_binary = buffer.getvalue()
            
            base64_encoded = base64.b64encode(image_binary)
            base64_string = base64_encoded.decode('utf-8')
            
            return base64_string
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return None

def save_images(dir,response_body):
    # Create directory for saving images if it doesn't exist
    os.makedirs(dir, exist_ok=True)
    image_paths = []
        
    # Save each image to a file and collect paths
    for i, base64_image in enumerate(response_body.get("images", [])):
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        rand_name = random_string_name()
        path = os.path.join(dir, f"{rand_name}.png")
        image.save(path)
        image_paths.append(path)
        
    return image_paths if image_paths else None



def generate_single_image(prompt, negative_prompt="", quality="premium", num_images=1, height=720, width=1280, seed=0, cfg_scale=6.5):
    """Generate image using Nova Canvas model"""
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText": negative_prompt
        } if negative_prompt else {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": int(num_images),  # Ensure integer
            "height": int(height),
            "width": int(width),
            "cfgScale": float(cfg_scale),
            "seed":  random.randint(0,858993459) if int(seed) == -1 else int(seed),
            "quality": quality
        }
    })
    
    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId='amazon.nova-canvas-v1:0',
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        # Save images using save_images function
        image_paths = save_images('generated_images', response_body)
        
        return image_paths if image_paths else None
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

def optimize_video_prompt(prompt,image_path):
    
    doc_bytes = load_system_prompts('./Amazon_Nova_Reel.pdf')
    model_id = 'us.amazon.nova-pro-v1:0'
    
    # base64_image = png_to_rgb_base64(image_path)
    image = Image.open(image_path)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    system = [{"text": SYSTEM_IMAGE_TEXT}]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "document": {
                        "format": "pdf",
                        "name": "DocumentPDFmessages",
                        "source": {"bytes": doc_bytes}
                    }
                },
                {"image": {"format": "png", "source": {"bytes": img_bytes}}},
                {"text": f"Please optimize: {prompt}"},
            ],
        }
    ]
        

    # Configure inference parameters
    inf_params = {"maxTokens": 512, "topP": 0.9, "temperature": 0.8}
    
    # Get response
    response = client.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inf_params
    )
    
    # Collect response text
    text = ""
    stream = response.get("stream")
    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                text += event["contentBlockDelta"]["delta"]["text"]
    
    optimized = parse_prompt(text)
    length = len(optimized)

    if length > MAX_PROMPT_LENGTH:
        return 'Error, Prompts too long'
    else :
        return optimized

def optimize_prompt(prompt, model_name, task):
    """
    Optimize user input prompt for Nova Canvas model using system prompt.
    
    Args:
        model_name: Name of the Nova model being used ('Nova Lite' or 'Nova Pro')
        input: User input prompt to be optimized
        task: Type of task ('image', 'video', or 'translate')
        
    Returns:
        tuple: (optimized_prompt, negative_prompt) where negative_prompt may be None
    """

    if task == 'image':
        system_prompt = SYSTEM_CANVAS
    elif task == 'video':
        system_prompt = SYSTEM_IMAGE_TEXT
    elif task == 'translate':
        system_prompt = SYSTEM_TRANSLATE
    else:
        system_prompt = ''

    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Invalid model name. Must be one of: {', '.join(MODEL_OPTIONS.keys())}")

    system = [{"text": system_prompt}]
    messages = [
        {
            "role": "user",
            "content": [{"text": f"Please {'translate' if task == 'translate' else 'optimize'}: {prompt}"}],
        }
    ]

    
    # Configure inference parameters
    inf_params = {"maxTokens": 512, "topP": 0.9, "temperature": 0.8}
    
    # Get response
    response = client.converse_stream(
        modelId=MODEL_OPTIONS[model_name],
        messages=messages,
        system=system,
        inferenceConfig=inf_params
    )
    
    # Collect response text
    text = ""
    stream = response.get("stream")
    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                text += event["contentBlockDelta"]["delta"]["text"]
    
    # Extract prompt and negative prompt based on task type
    if task == 'translate':
        # For translation, just return the response text directly since it's already formatted correctly
        return text.strip(), ""
    else:
        # For other tasks, extract both prompt and negative prompt
        prompt = parse_prompt(text, r"<prompt>(.*?)</prompt>")
        try:
            negative_prompt = parse_prompt(text, r"<negative_prompt>(.*?)</negative_prompt>")
        except ValueError:
            negative_prompt = ""
        print("Prompt :", prompt)
        print("negative_prompt :", negative_prompt)
        return prompt, negative_prompt


def inpainting_image(prompt, maskPrompt, path, negativePrompt='', seed=0, max_retries=3):
    model_id = 'amazon.nova-canvas-v1:0'
    
    # Read and encode image
    try:
        img = Image.open(path)
        width, height = img.size
        input_image = png_to_rgb_base64(path)
        if not input_image:
            raise ValueError("Failed to encode image")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

    # Verify all parameters are valid
    if not all([prompt, maskPrompt, input_image]):
        print("Missing required parameters")
        return None

    # Prepare request body
    request_body = {
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "text": prompt,
            "image": input_image,
            "maskPrompt": maskPrompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "height": height,
            "width": width,
            "cfgScale": 8.0,
            "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed)
        }
    }
    
    # Add negative prompt if provided
    if negativePrompt:
        request_body["inPaintingParams"]["negativeText"] = negativePrompt

    # Convert to JSON
    body = json.dumps(request_body)

    print(f"Inpainting request - Path:{path},Prompt: {prompt}, Mask Prompt: {maskPrompt}, Path: {path}, Negative Prompt: {negativePrompt}, Seed: {seed}")
    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept="application/json",
            )
            response_body = json.loads(response.get("body").read())
            
            # Save images
            image_paths = save_images('after_inpainting', response_body)
            print(f"Inpainting completed successfully. Generated image paths: {image_paths}")
            return image_paths if image_paths else None
            
        except Exception as e:
            print(f"Error during inpainting (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Max retries reached")
                return None


def background_remove(path):
    # Read image from file and encode it as base64 string.
    with open(path, "rb") as image_file:
        input_image = base64.b64encode(image_file.read()).decode('utf8')

    body = json.dumps({
            "taskType": "BACKGROUND_REMOVAL",
            "backgroundRemovalParams": {
                "image": input_image,
            }
        })
    response = bedrock_runtime.invoke_model(
        body=body, modelId='amazon.nova-canvas-v1:0', accept="application/json", contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    image_paths = save_images('background_remove',response_body)
    return image_paths


def outpainting_image(path, maskPrompt, prompt, negativePrompt='', seed=0):
    model_id = 'amazon.nova-canvas-v1:0'
    
    # Read and encode image
    try:
        img = Image.open(path)
        width, height = img.size
        input_image = png_to_rgb_base64(path)
        if not input_image:
            raise ValueError("Failed to encode image")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

    # Prepare request body
    request_body = {
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "text": prompt,
            "image": input_image,
            "maskPrompt": maskPrompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "height": height,
            "width": width,
            "cfgScale": 8.0,
            "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed)
        }
    }
    
    # Add negative prompt if provided
    if negativePrompt:
        request_body["outPaintingParams"]["negativeText"] = negativePrompt

    # Convert to JSON
    body = json.dumps(request_body)

    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
        )
        response_body = json.loads(response.get("body").read())
        
        # Save images
        image_paths = save_images('outpainting', response_body)
        return image_paths if image_paths else None
        
    except Exception as e:
        print(f"Error during outpainting: {str(e)}")
        return None

def download_video(s3_uri):
    """Download video from S3 to local storage"""
    try:
        s3_client = boto3.client('s3')
        
        if not s3_uri.startswith('s3://'):
            raise ValueError("Invalid S3 URI format")
        
        path_parts = s3_uri[5:].split('/', 1)
        bucket_name = path_parts[0]
        s3_key = path_parts[1]
        
        os.makedirs('videos', exist_ok=True)
        
        filename = f"{s3_key.split('/')[0]}.mp4"
        local_path = os.path.join('videos', filename)
        
        s3_client.download_file(bucket_name, s3_key, local_path)
        return local_path
        
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None



def generate_video(prompt, DEFAULT_BUCKET, path=None, seed=0):

    # Prepare model input
    model_input = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": prompt,
        },
        "videoGenerationConfig": {
            "durationSeconds": 6,
            "fps": 24,
            "dimension": "1280x720",
            "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed),
        },
    }
    
    # Add image if provided
    if path is not None:
        base64_image = png_to_rgb_base64(path)
        model_input['textToVideoParams']['images'] = [{
            "format": "png",
            "source": {
                "bytes": base64_image
            }
        }]
    
    # Start async video generation
    invocation = bedrock_runtime.start_async_invoke(
        modelId="amazon.nova-reel-v1:0",
        modelInput=model_input,
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": DEFAULT_BUCKET
            }
        }
    )
    
    # Wait for completion
    while True:
        response = bedrock_runtime.get_async_invoke(
            invocationArn=invocation['invocationArn']
        )
        status = response["status"]
        if status != 'InProgress':
            break
        time.sleep(5)
    
    # Download video
    output_uri = f"{response['outputDataConfig']['s3OutputDataConfig']['s3Uri']}/output.mp4"
    local_path = download_video(output_uri)
    
    return local_path


if __name__ == '__main__':
    pass
    # Example usage:
    # optimize_prompt('一只猫在草地上玩耍，追逐一只蝴蝶，不能在室内','Nova Pro')
    # generate_single_image("A playful cat in a sunlit meadow chasing a butterfly, vibrant colors, dynamic action, shallow depth of field, nature photography style")
    inpainting_image('a colorful snake sitting and eatting','the panda','generated_images/jmano0q9nvc7.png','human presence, artificial objects')
    # outpainting_image('/Users/zhenwez/Downloads/workshop-data/nova/ws/background_remove_p1ocogcj1xh3.png','the old man and the Reindeer-drawn sleigh','riding fast in the small village')
    # background_remove('/Users/zhenwez/Downloads/workshop-data/nova/ws/generated_images/generated_al0psm3ip0hq.png')
    # generate_video('the sleigh is running', DEFAULT_BUCKET, path='/Users/zhenwez/Downloads/workshop-data/nova/ws/generated_images/generated_ufwh1essi85j.png', seed=0)
    # inpainting_image('a black bear eating bamboo','panda','generated_images/jmano0q9nvc7.png','human presence, artificial objects')
    #  negativePrompt='human presence, artificial objects'
