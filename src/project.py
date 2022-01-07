from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import typing
from starlette.responses import JSONResponse


# BaseModel definitions
class Project(BaseModel):
    id: int = 0
    name: str = ""  
    created_date: Optional[datetime] 
    description: Optional[str] = ""
    data_file: Optional[str] = ""
    created_by: str = "" 
    model: Optional[str] = "Supervised"
    algorithms: Optional[List[str]] = []
    features: Optional[List[str]] = []
    label: Optional[List[str]] = []
    accuracy: Optional[typing.Dict[str,List[float]]] = {}
    
    
#
# Exceptions
#

# Exceptions % Exception Handlers
class NotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        