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
    