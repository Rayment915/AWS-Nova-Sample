import streamlit as st
import os
import logging
from nova import (
    optimize_prompt,
    generate_single_image,
    inpainting_image,
    background_remove,
    outpainting_image,
    generate_video,
    optimize_video_prompt
)
from config import DEFAULT_BUCKET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AWS Nova for image and video generation", layout="wide")
st.title("AWS Nova for image and video generation")

# Initialize session state for all variables
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = ""
if 'optimized_prompt' not in st.session_state:
    st.session_state.optimized_prompt = ""
if 'negative_prompt' not in st.session_state:
    st.session_state.negative_prompt = ""
if 'painting_optimized_prompt' not in st.session_state:
    st.session_state.painting_optimized_prompt = ""
if 'painting_negative_prompt' not in st.session_state:
    st.session_state.painting_negative_prompt = ""
if 'inpainting_new_prompt' not in st.session_state:
    st.session_state.inpainting_new_prompt = ""
if 'outpainting_new_prompt' not in st.session_state:
    st.session_state.outpainting_new_prompt = ""
if 'inpainting_mask_prompt' not in st.session_state:
    st.session_state.inpainting_mask_prompt = ""
if 'outpainting_mask_prompt' not in st.session_state:
    st.session_state.outpainting_mask_prompt = ""
if 'translated_mask_prompt' not in st.session_state:
    st.session_state.translated_mask_prompt = ""
if 'translated_negative_prompt' not in st.session_state:
    st.session_state.translated_negative_prompt = ""

# Create sidebar with tabs
tab = st.sidebar.radio("Navigation", ["Image Generation", "Image painting", "Video Generation"])

if tab == "Image Generation":
    # Initialize seed in session state if not present
    if 'seed' not in st.session_state:
        st.session_state.seed = -1
    
    # Create two columns for layout
    left_col, right_col = st.columns([0.5, 0.5])
    
    with right_col:
        st.subheader("Generation Parameters")
        quality = st.selectbox(
            "Quality",
            ["premium", "standard"],
            index=0  # premium as default
        )
        
        seed = st.slider(
            "Seed",
            min_value=-1,
            max_value=858993459,
            value=st.session_state.seed  # use session state value
        )
        st.session_state.seed = seed  # update session state
        
        st.subheader("API Schema Sample")
        st.code('''
{
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": string,
        "negativeText": string
    },
    "imageGenerationConfig": {
        "width": int,
        "height": int,
        "quality": "standard" | "premium",
        "cfgScale": float,
        "seed": int,
        "numberOfImages": int
    }
}
        ''', language="json")
    
    with left_col:
        # Prompt optimization section
        st.subheader("Prompt Optimization")
        user_prompt = st.text_input("User Input Prompt", value=st.session_state.user_prompt)
        st.session_state.user_prompt = user_prompt
        
        if st.button("Optimize Prompt", key="generate_prompt_btn"):
            if user_prompt:
                logger.info(f"Optimizing prompt: {user_prompt}")
                try:
                    optimized_prompt, negative_prompt = optimize_prompt(user_prompt, "Nova Pro",'image')
                    logger.info(f"Prompt optimized successfully. Optimized: {optimized_prompt}, Negative: {negative_prompt}")
                    st.session_state.optimized_prompt = optimized_prompt
                    st.session_state.negative_prompt = negative_prompt
                except Exception as e:
                    logger.error(f"Error optimizing prompt: {str(e)}")
                    st.error("Failed to optimize prompt")
        
        # Always show the optimized prompt fields
        st.text_area("Optimized Prompt by Nova", value=st.session_state.optimized_prompt, height=68)
        if st.session_state.negative_prompt:
            st.text_area("Negative Prompt", value=st.session_state.negative_prompt, height=68)
        
        # Generate new image section
        st.subheader("Generated Image")
        if st.button("Generate Image", key="generate_image_btn"):
            if st.session_state.optimized_prompt:
                logger.info(f"Generating image with prompt: {st.session_state.optimized_prompt}")
                try:
                    image_paths = generate_single_image(
                        st.session_state.optimized_prompt,
                        st.session_state.negative_prompt,
                        quality=quality,
                        seed=seed
                    )
                    if image_paths:
                        logger.info(f"Image generated successfully: {image_paths[0]}")
                        st.image(image_paths[0], caption="Generated Image")
                    else:
                        logger.warning("No image paths returned from generation")
                except Exception as e:
                    logger.error(f"Error generating image: {str(e)}")
                    st.error("Failed to generate image")
            else:
                # Show empty placeholder for the image
                st.empty()

elif tab == "Image painting":
    # Create three columns for the top section
    left_col, middle_col, right_col = st.columns([0.3, 0.35, 0.35])
    
    with left_col:
        # Painting mode selection
        st.markdown("### Select Painting Mode")
        painting_mode = st.selectbox(
            "Select Mode",
            ["inpainting", "background_remove", "outpainting"],
            key="painting_mode_select"
        )
        
        # Available directories
        st.markdown("### Select Directory")
        available_dirs = ["generated_images", "background_remove", "outpainting", "after_inpainting"]
        selected_dir = st.selectbox(
            "Select Directory",
            available_dirs,
            key="directory_select"
        )
        
        # Get list of images from selected directory
        st.markdown("### Select Image")
        selected_image_path = None
        if os.path.exists(selected_dir):
            image_files = [f for f in os.listdir(selected_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                selected_image = st.selectbox(
                    "Select Image",
                    image_files,
                    key="image_select"
                )
                selected_image_path = os.path.join(selected_dir, selected_image)
                
                # Display selected image
                if os.path.exists(selected_image_path):
                    st.image(selected_image_path, caption="Selected Image")
            else:
                logger.warning(f"No images found in directory: {selected_dir}")
                st.warning(f"No images found in {selected_dir} directory")
        else:
            logger.warning(f"Directory does not exist: {selected_dir}")
            st.warning(f"Directory {selected_dir} does not exist")
    
    if painting_mode == "inpainting":
        with middle_col:
            # Input fields with consistent height
            mask_prompt = st.text_input("Mask Prompt for Inpainting", value=st.session_state.inpainting_mask_prompt, key="mask_prompt_input")
            st.session_state.inpainting_mask_prompt = mask_prompt
            
            new_prompt = st.text_input("New Prompt for Inpainting", value=st.session_state.inpainting_new_prompt, key="new_prompt_input")
            st.session_state.inpainting_new_prompt = new_prompt
            
            negative_prompt = st.text_input("Negative Prompt", value=st.session_state.painting_negative_prompt, key="negative_prompt_input")
            st.session_state.painting_negative_prompt = negative_prompt
            
            # Translate button
            translate_clicked = st.button("Translate Prompt", key="translate_prompts_btn")
        
        with right_col:
            # Show translated prompts with consistent height
            st.text_input("Translated Mask Prompt", value=st.session_state.translated_mask_prompt, key="translated_mask_display")
            st.text_input("Translated New Prompt", value=st.session_state.painting_optimized_prompt, key="translated_new_display")
            st.text_input("Translated Negative Prompt", value=st.session_state.translated_negative_prompt, key="translated_negative_display")
        
        # Add horizontal line to separate sections
        st.markdown("---")
        st.markdown("### Painting Area")
        
        # Create two columns for the bottom section
        bottom_left, bottom_right = st.columns([0.2, 0.8])
        
        with bottom_left:
            inpainting_clicked = st.button("Apply Inpainting", key="inpainting_btn")
        
        with bottom_right:
            # Add placeholder for inpainting result
            result_placeholder = st.empty()
            result_placeholder.image("https://via.placeholder.com/400x200?text=Inpainting+Result", caption="Inpainting Result")
        
        # Handle translate button click
        if translate_clicked:
            try:
                # Translate all prompts at once
                if st.session_state.inpainting_mask_prompt:
                    logger.info(f"Translating mask prompt: {st.session_state.inpainting_mask_prompt}")
                    translated_prompt, _ = optimize_prompt(st.session_state.inpainting_mask_prompt, "Nova Pro", 'translate')
                    logger.info(f"Mask prompt translated successfully: {translated_prompt}")
                    st.session_state.translated_mask_prompt = translated_prompt
                
                if st.session_state.inpainting_new_prompt:
                    logger.info(f"Translating new prompt: {st.session_state.inpainting_new_prompt}")
                    translated_prompt, _ = optimize_prompt(st.session_state.inpainting_new_prompt, "Nova Pro", 'translate')
                    logger.info(f"New prompt translated successfully: {translated_prompt}")
                    st.session_state.painting_optimized_prompt = translated_prompt
                
                if st.session_state.painting_negative_prompt:
                    logger.info(f"Translating negative prompt: {st.session_state.painting_negative_prompt}")
                    translated_negative, _ = optimize_prompt(st.session_state.painting_negative_prompt, "Nova Pro", 'translate')
                    logger.info(f"Negative prompt translated successfully: {translated_negative}")
                    st.session_state.translated_negative_prompt = translated_negative
                
                # Rerun once after all translations are done
                st.rerun()
            except Exception as e:
                logger.error(f"Error translating prompts: {str(e)}")
                st.error("Failed to translate prompts")
        
        # Handle inpainting application
        if st.session_state.get('inpainting_btn', False):
            if selected_image_path and st.session_state.inpainting_mask_prompt and st.session_state.painting_optimized_prompt:
                logger.info(f"Applying inpainting to {selected_image_path}")
                try:
                    # Use the translated mask prompt if available, otherwise use the original mask prompt
                    mask_prompt_to_use = st.session_state.translated_mask_prompt if st.session_state.translated_mask_prompt else st.session_state.inpainting_mask_prompt
                    
                    # Log all input values for debugging
                    logger.info("Inpainting inputs:")
                    logger.info(f"- Selected image path: {selected_image_path}")
                    logger.info(f"- Mask prompt: {mask_prompt_to_use}")
                    logger.info(f"- Optimized prompt: {st.session_state.painting_optimized_prompt}")
                    logger.info(f"- Negative prompt: {st.session_state.painting_negative_prompt}")
                    
                    # Verify file exists
                    if not os.path.exists(selected_image_path):
                        raise FileNotFoundError(f"Image file not found: {selected_image_path}")
                    
                    # Only include negativePrompt if it's not empty
                    kwargs = {
                        'prompt': st.session_state.painting_optimized_prompt,
                        'maskPrompt': mask_prompt_to_use,
                        'path': selected_image_path
                    }
                    if st.session_state.painting_negative_prompt:
                        kwargs['negativePrompt'] = st.session_state.painting_negative_prompt
                    
                    logger.info(f"Calling inpainting_image with kwargs: {kwargs}")
                    result_paths = inpainting_image(**kwargs)
                    
                    if result_paths:
                        logger.info(f"Inpainting completed successfully: {result_paths[0]}")
                        if os.path.exists(result_paths[0]):
                            with middle_col:
                                result_placeholder.image(result_paths[0], caption="Inpainting Result")
                        else:
                            logger.error(f"Generated image file not found: {result_paths[0]}")
                            st.error("Generated image file not found")
                    else:
                        logger.warning("No result paths returned from inpainting")
                        st.error("Failed to generate inpainting result")
                except FileNotFoundError as e:
                    logger.error(f"File not found error: {str(e)}")
                    st.error(f"Image file not found: {str(e)}")
                except Exception as e:
                    logger.error(f"Error during inpainting: {type(e).__name__}: {str(e)}")
                    st.error(f"An error occurred during inpainting: {type(e).__name__}")
    
    elif painting_mode == "background_remove":
        # Add Remove Background button in left column
        with left_col:
            st.markdown("### Remove Background")
            if st.button("Remove Background", key="background_remove_btn"):
                if selected_image_path:
                    logger.info(f"Removing background from {selected_image_path}")
                    try:
                        result_paths = background_remove(selected_image_path)
                        if result_paths:
                            logger.info(f"Background removal completed successfully: {result_paths[0]}")
                            st.image(result_paths[0], caption="Background Removed")
                        else:
                            logger.warning("No result paths returned from background removal")
                    except Exception as e:
                        logger.error(f"Error during background removal: {str(e)}")
                        st.error("Failed to remove background")
        
    elif painting_mode == "outpainting":
        with middle_col:
            # Input fields with consistent height
            mask_prompt = st.text_input("Mask Prompt for Outpainting", value=st.session_state.outpainting_mask_prompt, key="mask_prompt_outpainting")
            st.session_state.outpainting_mask_prompt = mask_prompt
            
            new_prompt = st.text_input("New Prompt for Outpainting", value=st.session_state.outpainting_new_prompt, key="new_prompt_outpainting")
            st.session_state.outpainting_new_prompt = new_prompt
            
            negative_prompt = st.text_input("Negative Prompt", value=st.session_state.painting_negative_prompt, key="negative_prompt_outpainting")
            st.session_state.painting_negative_prompt = negative_prompt
            
            # Translate button
            translate_clicked = st.button("Translate Prompt", key="translate_outpainting_prompts_btn")
        
        with right_col:
            # Show translated prompts with consistent height
            st.text_input("Translated Mask Prompt", value=st.session_state.translated_mask_prompt, key="translated_mask_outpainting")
            st.text_input("Translated New Prompt", value=st.session_state.painting_optimized_prompt, key="translated_new_outpainting")
            st.text_input("Translated Negative Prompt", value=st.session_state.translated_negative_prompt, key="translated_negative_outpainting")
        
        # Add horizontal line to separate sections
        st.markdown("---")
        st.markdown("### Painting Area")
        
        # Create two columns for the bottom section
        bottom_left, bottom_right = st.columns([0.2, 0.8])
        
        with bottom_left:
            outpainting_clicked = st.button("Apply Outpainting", key="outpainting_btn")
        
        with bottom_right:
            # Add placeholder for outpainting result
            result_placeholder = st.empty()
            result_placeholder.image("https://via.placeholder.com/800x400?text=Outpainting+Result", caption="Outpainting Result")
        
        # Handle translate button click
        if translate_clicked:
            try:
                # Translate all prompts at once
                if st.session_state.outpainting_mask_prompt:
                    logger.info(f"Translating mask prompt: {st.session_state.outpainting_mask_prompt}")
                    translated_prompt, _ = optimize_prompt(st.session_state.outpainting_mask_prompt, "Nova Pro", 'translate')
                    logger.info(f"Mask prompt translated successfully: {translated_prompt}")
                    st.session_state.translated_mask_prompt = translated_prompt
                
                if st.session_state.outpainting_new_prompt:
                    logger.info(f"Translating new prompt: {st.session_state.outpainting_new_prompt}")
                    translated_prompt, _ = optimize_prompt(st.session_state.outpainting_new_prompt, "Nova Pro", 'image')
                    logger.info(f"New prompt translated successfully: {translated_prompt}")
                    st.session_state.painting_optimized_prompt = translated_prompt
                
                if st.session_state.painting_negative_prompt:
                    logger.info(f"Translating negative prompt: {st.session_state.painting_negative_prompt}")
                    translated_negative, _ = optimize_prompt(st.session_state.painting_negative_prompt, "Nova Pro", 'translate')
                    logger.info(f"Negative prompt translated successfully: {translated_negative}")
                    st.session_state.translated_negative_prompt = translated_negative
                
                # Rerun once after all translations are done
                st.rerun()
            except Exception as e:
                logger.error(f"Error translating prompts: {str(e)}")
                st.error("Failed to translate prompts")
        
        # Handle outpainting application
        if st.session_state.get('outpainting_btn', False):
            if selected_image_path and st.session_state.outpainting_mask_prompt and st.session_state.painting_optimized_prompt:
                logger.info(f"Applying outpainting to {selected_image_path}")
                try:
                    # Use the translated mask prompt if available, otherwise use the original mask prompt
                    mask_prompt_to_use = st.session_state.translated_mask_prompt if st.session_state.translated_mask_prompt else st.session_state.outpainting_mask_prompt
                    
                    # Log the values being used
                    logger.info(f"Using mask prompt: {mask_prompt_to_use}")
                    logger.info(f"Using optimized prompt: {st.session_state.painting_optimized_prompt}")
                    
                    # Only include negativePrompt if it's not empty
                    kwargs = {
                        'path': selected_image_path,
                        'maskPrompt': mask_prompt_to_use,
                        'prompt': st.session_state.painting_optimized_prompt
                    }
                    if st.session_state.painting_negative_prompt:
                        kwargs['negativePrompt'] = st.session_state.painting_negative_prompt
                    
                    result_paths = outpainting_image(**kwargs)
                    if result_paths:
                        logger.info(f"Outpainting completed successfully: {result_paths[0]}")
                        result_placeholder.image(result_paths[0], caption="Outpainting Result")
                    else:
                        logger.warning("No result paths returned from outpainting")
                        st.error("Failed to generate outpainting result")
                except Exception as e:
                    logger.error(f"Error during outpainting: {str(e)}")
                    st.error("Failed to apply outpainting")

elif tab == "Video Generation":
    st.subheader("Video Generation")
    
    # Directory selection
    available_dirs = ["generated_images", "background_remove", "outpainting", "after_inpainting"]
    selected_dir = st.selectbox(
        "Select Directory",
        available_dirs,
        key="video_directory_select"
    )
    
    # Image selection
    selected_image_path = None
    if os.path.exists(selected_dir):
        image_files = [f for f in os.listdir(selected_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            selected_image = st.selectbox(
                "Select Image",
                image_files,
                key="video_image_select"
            )
            selected_image_path = os.path.join(selected_dir, selected_image)
            
            # Display selected image
            if os.path.exists(selected_image_path):
                st.image(selected_image_path, caption="Selected Image")
        else:
            logger.warning(f"No images found in directory: {selected_dir}")
            st.warning(f"No images found in {selected_dir} directory")
    else:
        logger.warning(f"Directory does not exist: {selected_dir}")
        st.warning(f"Directory {selected_dir} does not exist")
    
    # Initialize video prompt in session state if not present
    if 'video_optimized_prompt' not in st.session_state:
        st.session_state.video_optimized_prompt = ""

    # Create columns for the prompt section
    prompt_col1, prompt_col2, prompt_col3 = st.columns([0.5, 0.15, 0.35])
    
    with prompt_col1:
        video_prompt = st.text_input("Enter Video Prompt", key="video_prompt_input")
    
    with prompt_col2:
        if st.button("Optimize Prompt", key="optimize_video_prompt_btn"):
            if video_prompt and selected_image_path:
                try:
                    optimized_prompt = optimize_video_prompt(video_prompt, selected_image_path)
                    st.session_state.video_optimized_prompt = optimized_prompt
                except Exception as e:
                    logger.error(f"Error optimizing video prompt: {str(e)}")
                    st.error("Failed to optimize video prompt")
            else:
                st.warning("Please enter a prompt and select an image first")
    
    # Show optimized prompt in the third column
    with prompt_col3:
        st.text_area("Optimized Video Prompt", value=st.session_state.video_optimized_prompt, height=100, key="optimized_prompt_display")
    
    # Generate video button
    if st.button("Generate Video", key="generate_video_btn"):
        if st.session_state.video_optimized_prompt:
            logger.info(f"Generating video with prompt: {st.session_state.video_optimized_prompt}")
            with st.spinner("Generating video... This may take a few minutes."):
                try:
                    video_path = generate_video(st.session_state.video_optimized_prompt, DEFAULT_BUCKET, selected_image_path if selected_image_path else None)
                    if video_path:
                        logger.info(f"Video generated successfully: {video_path}")
                        st.video(video_path)
                    else:
                        logger.warning("No video path returned from generation")
                        st.error("Failed to generate video")
                except Exception as e:
                    logger.error(f"Error generating video: {str(e)}")
                    st.error("Failed to generate video")
