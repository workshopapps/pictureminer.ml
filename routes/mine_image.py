from fastapi import APIRouter, UploadFile
from generator import captioner_generator
from io import BytesIO

router = APIRouter(
	prefix='', 
	tags=['Caption Generator']
)


@router.post('/caption-generator')
async def get_image_content(image: UploadFile):
    """
    Receives an image and returns the caption of the image
    """
    request_object_content = await image.read()

    # get the image content
    caption = captioner_generator.predict(BytesIO(request_object_content))

    return { 'text_description': caption }
    
    
@router.post('/check-for-prompt/{prompt}')
async def check_text_prompt(image: UploadFile, prompt: str):
    """
    Receives an image and prompt and returns true if the prompt exist in the image
    """
    request_object_content = await image.read()
    caption = captioner_generator.predict(BytesIO(request_object_content))
    check_result = captioner_generator.check_similarity(prompt, caption)
    
    return { 'prompt_exist': check_result,'caption': caption }