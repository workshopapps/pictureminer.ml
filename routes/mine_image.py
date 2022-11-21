from fastapi import APIRouter, UploadFile
from generator import captioner_generator

router = APIRouter(
	prefix='/api/microservice', 
	tags=['/api/microservice']
)


@router.post('/content')
async def get_image_content(image: UploadFile):
    """
    Receives an image and returns the caption of the image
    """
    request_object_content = await image.read()

    # get the image content
    caption = captioner_generator.predict(request_object_content)

    return { 'text_description': caption }
    