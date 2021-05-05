from typing import List
import logging

import requests
from fastapi import APIRouter, HTTPException, UploadFile, File, Request

from app.models.schema.label import Label
from app.ml.models.annotation import annotate as annotate_process


router = APIRouter()

# add filemode="w" to overwrite
logging.basicConfig(filename="sample.log", level=logging.INFO)


@router.post('/test_annotate', response_model=List[Label])
async def test_annotate(file: UploadFile = File(...)):

    try:
        labeled_data = annotate_process(file.file, test=True)
    except Exception as e:
        logging.error(f'[ERROR]{e}')
        print(f'[ERROR]{e}')
        raise HTTPException(status_code=400, detail=f'[ERROR]{e}')

    return labeled_data


@router.post('/annotate', response_model=List[Label])
async def annotate(request: Request):

    ip = request.client.host
    body = (await request.body()).decode()
    url = f'http://{ip}/{body}'
    logging.info(f'URL: [{url}]')
    try:
        response = requests.get(url)
        file = response.content.decode()
        logging.info(f'Preview: {file[:1000]}')
    except Exception as e:
        logging.error(f'Ошибка скачивания: {e}')
        print(f'Ошибка скачивания [{url}]')
        raise HTTPException(status_code=400, detail=f'Ошибка скачивания: {e}')

    try:
        labeled_data = annotate_process(file)
    except Exception as e:
        logging.error(f'[ERROR]{e}')
        print(f'[ERROR]{e}')
        raise HTTPException(status_code=400, detail=f'[ERROR]{e}')

    return labeled_data
