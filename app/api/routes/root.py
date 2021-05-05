from fastapi import APIRouter

from . import annotation

router = APIRouter()

router.include_router(annotation.router, prefix='/data', tags=['Data'])


@router.get('/ping')
async def ping():
    return 'pong'
