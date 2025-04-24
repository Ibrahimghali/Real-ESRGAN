import streamlit as st
import torch
from PIL import Image
import io
from RealESRGAN import RealESRGAN
import os

def main():
    # Set page title and description
    st.set_page_config(page_title="Real-ESRGAN Image Enhancer", layout="wide")
    
    st.title("Real-ESRGAN Image Enhancement")
    st.markdown("""
    This application enhances low-resolution images using the Real-ESRGAN deep learning model.
    Upload your image, click "Enhance", and see the magic happen!
    """)
    
    # Create directories if they don't exist
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image to enhance", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Create two columns for before/after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Enhance button
        enhance_pressed = st.button("✨ Enhance Image")
        
        if enhance_pressed:
            # Show spinner while processing
            with st.spinner("Enhancing image... This may take a few seconds."):
                try:
                    # Initialize model
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = RealESRGAN(device, scale=4)
                    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
                    
                    # Process image
                    sr_image = model.predict(image)
                    
                    # Display enhanced image
                    with col2:
                        st.subheader("Enhanced Image")
                        st.image(sr_image, use_column_width=True)
                    
                    # Add download button
                    buf = io.BytesIO()
                    sr_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download Enhanced Image",
                        data=byte_im,
                        file_name="enhanced_image.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during enhancement: {str(e)}")
    
    # Information section
    with st.expander("About Real-ESRGAN"):
        st.markdown("""
        **Real-ESRGAN** is a powerful image enhancement model that can:
        - Increase resolution of images
        - Reduce noise and artifacts
        - Improve overall image clarity
        
        The model works best with photos, illustrations, and general images. Results may vary depending on 
        the input image quality and content.
        """)

if __name__ == '__main__':
    main()